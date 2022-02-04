import csv
import json
import requests

from requests.auth import HTTPBasicAuth


class RoutePaths:
    endpoint = "http://localhost:5000"
    user = "/users{subpath}"
    standup = "/standups{subpath}"


class BaseClient:
    def __init__(self, endpoint: str, route: str):
        self.endpoint = endpoint
        self.route = route
        self.auth = HTTPBasicAuth('standupbento', 'standupbento')


class SendGetClient(BaseClient):
    def __init__(self, endpoint, route):
        super().__init__(endpoint, route)


    def req(self):
        """
        req method sends a get request
        """
        url = url = f"{self.endpoint}{self.route}"
        try:
            res = requests.get(url, auth=self.auth)
            return res.json()
        except Exception as ex:
            raise ex


class SendPostClient(BaseClient):
    def __init__(self, endpoint, route, data=None, filepath=None):
        super().__init__(endpoint, route)
        self.data = data
        self.filepath = filepath


    def read_file(self):
        """
        read_file method reads file type json/csv
        """
        with open(self.filepath, 'r', encoding='utf-8-sig') as file:
            if self.filepath[-4:].lower() == "json":
                data = json.load(file)
                return data
            csv_reader = csv.DictReader(file)
            return list(csv_reader)


    def req(self):
        """
        req method sends a post request
        """
        url = f"{self.endpoint}{self.route}"
        try:
            if self.filepath is not None:
                res = requests.post(
                    url, json=self.read_file(), auth=self.auth)
                return res
            res = requests.post(
                url, json=self.data, auth=self.auth)
            return res
        except FileNotFoundError as ex:
            raise ex
        except Exception as ex:
            raise ex


def populate_data():
    """
    sends request to standupman endpoint to populate the mongodb with data
    """
    # generate user
    SendPostClient(RoutePaths.endpoint,
                   RoutePaths.user.format(subpath="/register"),
                   filepath="client/data/user.json").req()
    # generate standup
    SendPostClient(RoutePaths.endpoint,
                   RoutePaths.standup.format(subpath="/new"),
                   filepath="client/data/standup.json").req()
    # generate standup responses
    user_id = SendGetClient(RoutePaths.endpoint,
                            RoutePaths.user.format(subpath="/")).req()
    standup_id = SendGetClient(
        RoutePaths.endpoint, RoutePaths.standup.format(subpath="/")).req()
    print(user_id['users'][0]['_id'])
    print(standup_id['standups'][0]['_id'])

    csv_reader = SendPostClient(RoutePaths.endpoint, RoutePaths.standup.format(
        subpath="/complete"), filepath="client/data/standup_responses.csv").read_file()
    for row in csv_reader:
        standup_responses = {
            "standup_update": {
                "standup_id": standup_id['standups'][0]['_id'],
                "user_id": user_id['users'][0]['_id'],
                "responseTime": row["responseTime"],
                "answers": {
                    "answer_1": {
                        "question_id": "question_1",
                        "response": row["standupResponse"]
                    }
                }
            }
        }
        SendPostClient(RoutePaths.endpoint, RoutePaths.standup.format(
            subpath="/complete"), data=standup_responses).req()


def get_standup_responses():
    """
    sends a request to get data from mongodb
    """
    data = SendGetClient(RoutePaths.endpoint,
                         RoutePaths.standup.format(subpath="/responses")).req()
    return [response["answers"]["answer_1"]["response"] for response in data["standUpResponses"]]


if __name__ == "__main__":
    populate_data()

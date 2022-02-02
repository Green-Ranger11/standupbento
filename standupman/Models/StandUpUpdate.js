import mongoose from 'mongoose'
const Schema = mongoose.Schema

const StandUpUpdateSchema = new Schema({
    standup_id: {
        type: String,
        minlength: [4, 'standup id too short'],
        required: [true, 'standup_id not present.']
    },
    user_id: {
        type: String,
        minlength: [4, 'user id too short'],
        required: [true, 'user_id not present.']
    },
    responseTime: {
        type: Date,
        required: [true, 'responseTime must be present']
    },
    answers: {
        type: Object,
        validate: {
            validator: function (input) {
                let answers = input;
                let keys = Object.keys(answers);
                if (keys.length < 1) {
                    return false;
                }

                for(let i = 0; i < keys.length; i++) {
                  if(Object.keys(answers[keys[i]]).length < 2) {
                      return false;
                  }
                  if(answers[keys[i]].question_id === undefined || answers[keys[i]].question_id === "") {
                      return false;
                  }
                  if(answers[keys[i]].response === undefined || answers[keys[i]].response ===  "") {
                      return false;
                  }
                }
                
                return true;
            
            },
            message: 'A minimum of one answer must be present with the form { answer_1: {question_id: question_1, response: "My answer 1"}, answer_2: {question_id: question_2, response: "My answer 2"}}',
        },
        required: [true, 'answers object not present']
    },
    user_id: String
});

const StandUpUpdate = mongoose.model('StandUpUpdate', StandUpUpdateSchema);
export default StandUpUpdate;
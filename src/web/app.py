# Necessary imports
# from pipelines import pipeline
# from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline as pl
import pickle
from flask import Flask, json, jsonify, request, render_template, Response, send_from_directory, session
# from keyphrase_extraction import get_keyphrases
# from sentence_sim import *
import os
import shutil

# Flask App
app = Flask(__name__)
app.secret_key = 'aitutor'

# ==================================== Initialize global variables ============================================
course_root = './Courses/'

# ======================= Import models to be used by the app =================================================
# print("Loading models. Please wait...")
# models = ["basic_model", "t5-base-qg-hl", "multitask-qa-qg", "t5-base-qa-qg-hl", "e2e-qg", "t5-base-e2e-qg"]
# nlp_small = pipeline("question-generation", model="valhalla/t5-small-qa-qg-hl")

# Import tokenizer and model (Download model) Used for question generation based on context and answer.
# tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
# model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")#.to('cuda:0')

# ============================================================================================================

# Generate questions using the small t5 model
# ===========================================
def gen_question_small(topic_text):
    questions = nlp_small(topic_text)
    return questions

# Answer aware question generation
def get_question(answer, context, max_length=64):

    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = tokenizer([input_text], return_tensors='pt', truncation=True, max_length=1024)#.to('cuda:0')

    output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=max_length).to('cpu')

    return tokenizer.decode(output[0])

# Function to remove duplicates from the questions set
def remove_duplicates(questSet):
    pass



# ================================ ALL FUNCTIONS DEFINED ABOVE ===================================================

# ================================ SERVER CODE BELOW =============================================================

# Root entry point
# =============================================================
@app.route("/")
def index():
    return jsonify({"Message": "Welcome to AI Tutor"})

# Show Courses route
# =============================================================
@app.route("/showcourses/", methods=["GET"])
def list_all_courses():
    courses = []
    for c in os.listdir(course_root):
        courses.append(c)

    return jsonify(courses)

# Show topics route
# =============================================================
@app.route("/showtopics/", methods=["GET"])
def list_all_topics():
    course_name = request.args.get("course")
    print(course_name)
    topics = []
    if(course_name is not None):
        for t in os.listdir(course_root + course_name):
            topics.append(t)
    else:
        return("Course name not specified in request!!")

    return jsonify(topics)

# Get contents for a topic
# =============================================================
@app.route("/get_topic/", methods=["GET"])
def get_topic_text():
    course_name = request.args.get("course")
    topic_name = request.args.get("topic")
    
    topic_text = []
    if(course_name is not None and topic_name is not None):
        for tf in os.listdir(course_root + course_name + "/" + topic_name):
            file = open(course_root + course_name + '/' + topic_name + "/" + tf, 'r')
            doc = file.read()

            topic_text.append(doc)
    else:
        return("Course name and/or topic name not specified in request!!")

    return jsonify(topic_text)

# Generate questions and answers route for a topic
# =============================================================
@app.route("/gen_qa/", methods=["GET"])
def generate_qa():
    course_name = request.args.get("course")
    topic_name = request.args.get("topic")
    force = bool(request.args.get("force"))
    
    if(force is None):
        force=False

    topic_text = []
    if(course_name is not None and topic_name is not None):
        topic_dir = course_root + course_name +"/" + topic_name
        for tf in os.listdir(course_root + course_name + "/" + topic_name):
            file = open(course_root + course_name + '/' + topic_name + "/" + tf, 'r')
            doc = file.read()

            topic_text.append(doc)
    else:
        return("Course name and/or topic name not specified in request!!")

    # Check for existence of generated questions
    qafile = f"{course_root}{course_name}{'/'}{topic_name}_all_qa"
    if(os.path.exists(qafile) and force!=True):
        all_qa = pickle.load(open(qafile, 'rb'))
        print("Previous File Loaded")
    else:
        # Generate questions using the QG model
        print("Generating questions and answers....")
        all_questions = []
        all_answers = []

        # Generate questions for all sources
        for doc in topic_text:
            answers = []
            quest = gen_question_small(doc)

            for q in quest:
                all_questions.append(q['question'])
                all_answers.append(q['answer'])

            # Generate answer-aware questions using the answers from the previous model
            for q in quest:
                answers.append(q["answer"])

            for a in answers:
                all_questions.append(get_question(a, doc)[10:])
                all_answers.append(a)

            ans = get_keyphrases(doc)

            new_qa = []
            for a in ans:
                all_questions.append(get_question(a, doc)[10:])
                all_answers.append(a)

        # Delete redundant questions
        q_emb = get_quest_embeddings(all_questions)

        num_quest = q_emb.shape[0]

        sim_quest = []
        for q1 in range(num_quest):
            for q2 in range(q1+1, num_quest):
                sim = F.cosine_similarity(q_emb[q1].unsqueeze(dim=0),q_emb[q2].unsqueeze(dim=0))
                if(sim>0.95):
                    sim_quest.append([q1, q2, sim])

        # Store generated questions
        all_qa = []
        for q,a in zip(all_questions, all_answers):
            all_qa.append({'Quest': q, 'Ans': a})

        # Store questions in a file
        pickle.dump(all_qa, open(qafile, 'wb'))
        print("QA stored in file")
        print(all_qa)

    # Return the generated question answers list
    return jsonify(all_qa)

# Get specified content for the topic
# =============================================================
@app.route("/get_one_topic/", methods=["GET"])
def get_one_topic_text():
    course_name = request.args.get("course")
    topic_name = request.args.get("topic")
    topic_num = request.args.get("num")
    
    topic_text = []
    if(course_name is not None and topic_name is not None):
        for tf in os.listdir(course_root + course_name + "/" + topic_name):
            file = open(course_root + course_name + '/' + topic_name + "/" + tf, 'r')
            doc = file.read()

            topic_text.append(doc)
    else:
        return("Course name and/or topic name not specified in request!!")

    if(topic_num<len(topic_text)):
        return jsonify(topic_text[topic_num])
    else:
        return jsonify(topic_text[0])

# Add additional content to a topic
# ===============================================================
@app.route("/add_text_topic/", methods=["POST"])
def add_one_topic_text():
    course_name = request.form["course"]
    topic_name = request.form["topic"]
    topic_txt = request.form["topic_txt"]

    pth = course_root + course_name + '/' + topic_name

    fname = len(os.listdir(pth)) + ".txt"
    file = open(pth + "/" + fname, "w")
    file.write(topic_txt)
    file.close()

    return ({"success": True})

# show all previously generated questions and answers
# ================================================================
@app.route("/get_qa/", methods=["GET"])
def get_qa():
    course_name = request.args.get("course")
    topic_name = request.args.get("topic")

    topic_text = []
    if(course_name is not None and topic_name is not None):
        topic_dir = course_root + course_name +"/" + topic_name
        # Check for existence of generated questions
        qafile = f"{course_root}{course_name}{'/'}{topic_name}_all_qa"
        if(os.path.exists(qafile)):
            all_qa = pickle.load(open(qafile, 'rb'))
            print("Previous File Loaded") 
            return(jsonify(all_qa))
        else:
            return({"success": False, "error": "QA file not found. Generate questions first and then try again."})
    else:
        return({"success": False, "error": "Course name and/or topic name not specified in request!!"})

    
# Delete selected course
# ================================================================
@app.route("/delete_course/", methods=["GET"])
def delete_course():
    course_name = request.args.get("course")

    # Delete the course folder
    try:
        shutil.rmtree(course_root + course_name)
    except Exception as e:
            return ({"success": False, "error": e})

    return ({"success": True})

# Add new course
# ================================================================
@app.route("/add_course/", methods=["GET"])
def add_course():
    course_name = request.args.get("course")

    # add new course
    if(course_name is not None):
        # add a new folder for the new course
        try:
            os.makedirs(course_root + course_name)
            print("Course added!")
        except Exception as e:
            return ({"success": False, "error": e})
    else:
        return ({"success": False, "error": "Course name not provided."})

    return ({"success": True})

# Add topic to course
# ================================================================
@app.route("/add_topic/", methods=["POST"])
def add_topic():
    course_name = request.form["course"]
    topic_name = request.form["topic"]
    topic_txt = request.form["topic_txt"]

    # add a new topic to the existing course
    if(course_name is not None and topic_name is not None and topic_txt is not None):
        # add file to the specified topic
        try:
            os.makedirs(course_root + course_name + "/" + topic_name)
            print("Topic added!")
            # add text to the topic as well.
            pth = course_root + course_name + '/' + topic_name

            fname = len(os.listdir(pth)) + ".txt"
            file = open(pth + "/" + fname, "w")
            file.write(topic_txt)
            file.close()
        except Exception as e:
            return ({"success": False, "error": e})

    return ({"success": True})

# Edit topic in course
# ================================================================
@app.route("/edit_topic/", methods=["POST"])
def edit_topic():
    course_name = request.form["course"]
    topic_name = request.form["topic"]
    topic_txt = request.form["topic_txt"]
    topic_id = request.form["topic_id"]

    # Code here

    return ({"success": True})


# Add QA to a topic in a course
# ================================================================
@app.route("/add_qa/", methods=["POST"])
def add_qa():
    course_name = request.form["course"]
    topic_name = request.form["topic"]
    quest = request.form["quest"]
    ans = request.form["ans"]

    # Add the new question-answer to the existing QA file
    if(course_name is not None and topic_name is not None):
        # Check for existence of generated questions
        qafile = f"{course_root}{course_name}{'/'}{topic_name}_all_qa"
        if(os.path.exists(qafile)):
            all_qa = pickle.load(open(qafile, 'rb'))
            print("Previous File Loaded") 

            print("Adding the new question...")
            all_qa.append({"quest": quest, "ans": ans})

            print("Saving file...")
            pickle.dump(all_qa, open(qafile, 'wb'))

            return ({"success": True})
        else:
            return ({"success": False, "error": "QA file does not exist. First generate questions for this topic and then try again."})
    else:
        return({"success": False, "error": "Course name and/or topic name not specified in request!!"})
    


# Delete QA from a topic in a course
# ================================================================
@app.route("/delete_qa/", methods=["GET"])
def delete_qa():
    course_name = request.form["course"]
    topic_name = request.form["topic"]
    quest = request.form["quest"]
    ans = request.form["ans"]

    # To be finalized when the interface design is decided

    return ({"success": True})

# Edit QA to a topic in a course
# ================================================================
@app.route("/edit_qa/", methods=["POST"])
def edit_qa():
    course_name = request.form["course"]
    topic_name = request.form["topic"]
    quest = request.form["quest"]
    ans = request.form["ans"]

    # To be finalized on the basis of UI

    return ({"success": True})

# Edit QA to a topic in a course
# ================================================================
@app.route("/start_qa_session/", methods=["GET"])
def start_qa_session():
    course_name = request.form["course"]
    topic_name = request.form["topic"]


    # Code here [Recommeder Algorithm] Enters the student into chatbot mode

    return ({"success": True})


# Server Entry Point
# ====================
if __name__ == '__main__':
    print(f"Server Running at Port No. 8080...")
    # serve(app, host='0.0.0.0', port=8080, threads=16, _quiet=False)
    app.run(host="0.0.0.0", port=8080, debug=False)

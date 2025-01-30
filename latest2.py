import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from groq import Groq
import re
import pdfplumber
import requests
import json

# Configuration
GROQ_API_KEY = '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'
MODEL_NAME = "llama-3.3-70b-versatile"
PERSIST_DIRECTORY = 'db'

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'resume_analyses' not in st.session_state:
    st.session_state.resume_analyses = {}
if 'current_document' not in st.session_state:
    st.session_state.current_document = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_triggers' not in st.session_state:
    st.session_state.selected_triggers = []
if 'custom_triggers' not in st.session_state:
    st.session_state.custom_triggers = []
if 'trigger_faqs' not in st.session_state:
    st.session_state.trigger_faqs = {}

TRIGGER_OPTIONS = [
    "Time Gap Between Jobs",
    "Domain Switch",
    "Same Role for Long Duration"
]

COMMON_QUESTIONS = {
    "Time Gap Between Jobs": [
        "Can you explain the reason for the gap between jobs?",
        "What were you doing during this time?",
        "Did you pursue any personal or professional development during the gap?"
    ],
    "Domain Switch": [
        "What motivated you to switch domains?",
        "What skills from your previous domain are transferable to your new domain?",
        "Did you face any challenges while transitioning between domains?"
    ],
    "Same Role for Long Duration": [
        "What kept you in the same role for such a long period?",
        "Did you take on additional responsibilities during this time?",
        "What achievements or growth did you experience in this role?"
    ]
}

def extract_resume_details(text, job_description):
    """
    Extract key details from resume text based on job description using Groq
    """
    chat_model = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
    )

    prompt = f"""
    Given the following resume text and job description, extract and organize the following key details:
    1. Contact Information
    2. Years of Experience
    3. Key Skills (especially those matching the job description)
    4. Previous Companies and Roles
    5. Education
    6. Certifications
    7. Job Description Match Score (analyze how well the resume matches the requirements)

    Additionally, analyze the following trigger responses:
    - Detect gaps between jobs (e.g., more than 6 months between leaving one company and starting another). Clearly specify the duration and timeframes.
    - Identify any major domain switches between roles (e.g., switching from software development to marketing). Highlight the specific roles and domains involved.
    - Highlight if the candidate worked in the same role or company for more than 5 years. Include the role, company, and duration.

    If a scenario does not occur (e.g., no domain switch or gap), explicitly mention it to avoid ambiguity.

    Based on these trigger responses, generate questions that can be asked during an interview.

    Job Description:
    {job_description}

    Resume Text:
    {text}

    Provide the information in a structured format, including:
    - Key details (1-7 above)
    - Trigger analyses (explicitly stating the presence or absence of each scenario)
    - Suggested interview questions based on triggers.
    """

    response = chat_model.invoke(prompt)
    return response.content

def analyze_custom_triggers(text, custom_triggers):
    """
    Analyze custom triggers based on their description prompts and generate FAQs if triggers are found.
    Returns a dictionary of trigger names mapped to their FAQs or detection status.
    """
    chat_model = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
    )

    trigger_faqs = {}

    for trigger in custom_triggers:
        name = trigger['name']
        desc = trigger['desc']

        # First, explicitly check if the trigger condition exists
        detection_prompt = f"""
        Carefully analyze if the following condition exists in the resume text:
        
        Condition to check: {desc}
        
        Resume Text:
        {text}
        
        Respond ONLY with:
        "FOUND" - if you can find clear evidence of this condition in the resume
        "NOT_FOUND" - if you cannot find clear evidence of this condition
        
        Be strict in your assessment - only respond with FOUND if there is explicit evidence in the resume text.
        """

        detection_response = chat_model.invoke(detection_prompt).content.strip().upper()

        if "NOT_FOUND" in detection_response:
            trigger_faqs[name] = []
            continue

        if "FOUND" in detection_response:
            # Only proceed with question generation if trigger was found
            question_prompt = f"""
            Given the following trigger condition that was found in the resume:
            {desc}
            
            Generate 3-5 specific interview questions about this condition as it appears in the resume text:
            {text}
            
            Requirements:
            1. Questions must be directly related to evidence found in the resume
            2. Questions should help explore this specific situation
            3. Questions must be based on actual content from the resume, not hypotheticals
            
            Format: Start each question with "Q: "
            """
            
            question_response = chat_model.invoke(question_prompt).content.strip()
            questions = [q.strip() for q in question_response.split('\n') if q.strip().startswith('Q:')]
            
            trigger_faqs[name] = questions if questions else []
        else:
            trigger_faqs[name] = []

    return trigger_faqs

def reanalyze_custom_triggers(document_name):
    """
    Reanalyze a document with current custom triggers
    """
    try:
        # Get the document text from the stored vector store
        vector_store = Chroma(
            persist_directory=f"{PERSIST_DIRECTORY}/{document_name}",
            embedding_function=embeddings
        )
        
        # Get all texts from the vector store
        documents = vector_store.get()
        if not documents or 'documents' not in documents:
            st.error("Could not retrieve document text for reanalysis")
            return False
            
        # Reconstruct the document text
        text = " ".join(documents['documents'])
        
        # Only proceed if we have both text and triggers to analyze
        if not text or not st.session_state.custom_triggers:
            return False
            
        # Use the same analyze_custom_triggers function for consistency
        new_trigger_faqs = analyze_custom_triggers(text, st.session_state.custom_triggers)
        
        # Update the stored results
        if document_name not in st.session_state.trigger_faqs:
            st.session_state.trigger_faqs[document_name] = {}
            
        # Replace all custom trigger results with new analysis
        st.session_state.trigger_faqs[document_name] = new_trigger_faqs
                
        return True
        
    except Exception as e:
        st.error(f"Error reanalyzing document: {str(e)}")
        return False

def process_document(file, job_description):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Analyze resume details
        resume_analysis = extract_resume_details(text, job_description)
        
        # Analyze custom triggers if any exist
        if st.session_state.custom_triggers:
            trigger_faqs = analyze_custom_triggers(text, st.session_state.custom_triggers)
        else:
            trigger_faqs = {}

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)

        # Create vector store
        vector_store = Chroma.from_texts(
            chunks,
            embeddings,
            persist_directory=f"{PERSIST_DIRECTORY}/{file.name}"
        )

        # Store document info
        st.session_state.documents[file.name] = {
            'name': file.name,
            'path': f"{PERSIST_DIRECTORY}/{file.name}",
            'analysis': resume_analysis
        }
        st.session_state.resume_analyses[file.name] = resume_analysis
        st.session_state.trigger_faqs[file.name] = trigger_faqs
        return True

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False

def display_common_questions():
    selected_triggers = st.sidebar.multiselect("Select Triggers", TRIGGER_OPTIONS)
    st.session_state.selected_triggers = selected_triggers

    if selected_triggers:
        st.header("Common Questions Based on Selected Triggers")
        for trigger in selected_triggers:
            st.subheader(trigger)
            for question in COMMON_QUESTIONS.get(trigger, []):
                st.markdown(f"- {question}")

def chat_with_document(document_name, question):
    try:
        # Load vector store
        vector_store = Chroma(
            persist_directory=f"{PERSIST_DIRECTORY}/{document_name}",
            embedding_function=embeddings
        )

        # Initialize Groq chat model
        chat_model = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME
        )

        # Create conversation chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )

        # Get response
        response = qa_chain({
            "question": question,
            "chat_history": st.session_state.chat_history
        })

        return response['answer']

    except Exception as e:
        st.error(f"Error processing chat: {str(e)}")
        return None

def display_custom_trigger_results(trigger_faqs):
    st.header("Custom Trigger Analysis")
    
    if not trigger_faqs:
        st.info("No custom triggers analyzed yet. Add custom triggers in the sidebar to analyze the CV for specific conditions.")
        return

    for trigger_name, faqs in trigger_faqs.items():
        st.subheader(f"Trigger: {trigger_name}")
        if not faqs:  # If empty list, trigger wasn't found
            st.write("⚠️ This trigger condition was not found in the CV")
        else:
            for faq in faqs:
                st.markdown(f"- {faq}")


def extract_key_info(data):
    try:
        prompt = f"""
        Please extract the following information from the resume text below:

        1. Full Name
        2. Industry
        3. Job Title
        4. Current Location
        5. Job Location
        6. Education Details
        7. Job Start Date
        8. Job End Date (if applicable)
        9. Skills
        10. Interests
        11. Experience
        12. Current Company Name

        Resume Text:
        {data}

        Provide ONLY the extracted information in the following JSON format without any gap or space and with no additional commentary or explanations:

        {{
            "full_name": "<extracted_full_name>",
            "industry": "<extracted_industry>",
            "job_title": "<extracted_job_title>",
            "current_location": "<extracted_current_location>",
            "job_location": "<extracted_job_location>",
            "education": "<extracted_education_details>",
            "job_start_date": "<extracted_job_start_date>",
            "job_end_date": "<extracted_job_end_date>",
            "skills": ["<skill_1>", "<skill_2>", ...],
            "interests": ["<interest_1>", "<interest_2>", ...],
            "experience": ["<experience_1>", "<experience_2>", ...],
            "current_company": "<extracted_current_company>"
        }}
        """

        client = Groq(api_key="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")

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile"
        )
        message = chat_completion.choices[0].message.content
        return message
    except Exception as e:
        return{
            "error": f"An error occurred: {str(e)}"
        }

def compare_data(data1, data2):
    try:
        prompt = f"""
        You are an advanced LLM tasked with verifying the correctness of information in data2 by comparing it with data1. For each field, check the accuracy of the information provided in data2 and calculate a percentage match.
        Please do not generate your comments and explanations.
        Provide an individual score for each field and an overall percentage score based on the following criteria:

        1. **Full Name**: Check if the `full_name` fields in data1 and data2 match exactly (case-insensitive).
        2. **Occupation**: Compare the `occupation` field in data1 with the `job_title` field in data2 using semantic similarity.
        3. **Headline**: Compare the `headline` in data1 with any relevant title or description in data2 for semantic similarity.
        4. **Country**: Verify if the `country_full_name` field in data1 matches the `current_location` or `job_location` field in data2.
        5. **City**: Check if the `city` field in data1 is included in the `current_location` field in data2.
        6. **State**: Verify if the `state` field in data1 matches any part of the `current_location` or `job_location` field in data2.
        7. **Experiences**: Compare the `experiences` list in data1 with the `experience` list in data2 for similarity. Check company names, job titles, and descriptions.
        8. **Education**: Compare the `education` fields in data1 and data2, including degree name, field of study, school, and date ranges.
        9. **Skills**: Calculate the percentage overlap between the `skills` lists in data1 and data2.

        Use the following weights for the overall trueness score:
        - Full Name: 10%
        - Occupation: 10%
        - Headline: 10%
        - Country: 10%
        - City: 5%
        - State: 5%
        - Experiences: 30%
        - Education: 10%
        - Skills: 10%

        Provide the response in the following JSON format:
        {{
            "full_name_match_percentage": <value>,
            "occupation_match_percentage": <value>,
            "headline_match_percentage": <value>,
            "country_match_percentage": <value>,
            "city_match_percentage": <value>,
            "state_match_percentage": <value>,
            "experiences_match_percentage": <value>,
            "education_match_percentage": <value>,
            "skills_match_percentage": <value>,
            "overall_trueness_percentage": <value>
        }}

        Use the data below:
        data1:
        {data1}

        data2:
        {data2}
        """

        client = Groq(api_key="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")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile"
        )

        message = chat_completion.choices[0].message.content
        return message
    except Exception as e:
        return {
            "error": f"An error occurred: {str(e)}"
        }

def extract_data(data, keys):
    def recursive_extract(data, key, path=""):
        if isinstance(data, dict):
            for k, v in data.items():
                if k == key:
                    yield (path + k, v)
                if isinstance(v, (dict, list)):
                    yield from recursive_extract(v, key, path + k + ".")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                yield from recursive_extract(item, key, path + f"[{i}].")

    extracted = {}
    for key in keys:
        matches = list(recursive_extract(data, key))
        extracted[key] = matches if len(matches) > 1 else (matches[0][1] if matches else None)

    return extracted

def extract_linkedin_url(text):
    linkedin_pattern = r'https?://(?:www\.)?linkedin\.com/in/[\w-]+/?'
    match = re.search(linkedin_pattern, text)
    return match.group() if match else None



# Streamlit UI
st.title("Resume Parser and Analyzer")

# Sidebar for document upload and selection
with st.sidebar:
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    job_description = st.text_area("Enter Job Description")

    if uploaded_file and job_description:
        if st.button("Process Resume"):
            with st.spinner("Processing resume..."):
                if process_document(uploaded_file, job_description):
                    st.success("Resume processed successfully!")
                    st.session_state.current_document = uploaded_file.name

    st.header("Select Document")
    if st.session_state.documents:
        selected_document = st.selectbox(
            "Choose a document to analyze",
            options=list(st.session_state.documents.keys())
        )
        if selected_document:
            st.session_state.current_document = selected_document

    # Custom triggers section in sidebar
    st.header("Custom Triggers")
    with st.form("add_trigger_form"):
        trigger_name = st.text_input("Trigger Name")
        trigger_desc = st.text_area("Trigger Description (What to look for in the CV)")
        submit_button = st.form_submit_button("Add Trigger")
        
        if submit_button and trigger_name and trigger_desc:
            # Add new trigger
            st.session_state.custom_triggers.append({
                "name": trigger_name,
                "desc": trigger_desc
            })
            
            # Reanalyze current document if one is selected
            if st.session_state.current_document:
                with st.spinner(f"Analyzing CV for trigger: {trigger_name}..."):
                    if reanalyze_custom_triggers(st.session_state.current_document):
                        st.success(f"CV analyzed for trigger: {trigger_name}")
                    else:
                        st.error("Failed to analyze CV for the new trigger")
            else:
                st.success(f"Added trigger '{trigger_name}'. Upload a CV to analyze.")

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
       extracted_text = ''.join([page.extract_text() for page in pdf.pages])
       linkedin_url = extract_linkedin_url(extracted_text)

    if linkedin_url:
        api_key = 'xrhZ_CIZWGPnTIdQwwSFVQ'
        headers = {'Authorization': 'Bearer ' + api_key}
        api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
        params = {
            'linkedin_profile_url': linkedin_url,
            'extra': 'include',
            'github_profile_id': 'include',
            'personal_contact_number': 'include',
            'personal_email': 'include',
            'inferred_salary': 'include',
            'skills': 'include',
            'use_cache': 'if-present',
            'fallback_to_cache': 'on-error',
        }
        response = requests.get(api_endpoint,
                                params=params,
                                headers=headers)

        if response.status_code == 200:
                linkedin_data = response.json()
                with open("linkedin_data.json", "w") as file:
                    json.dump(linkedin_data, file, indent=4)

        with open("linkedin_data.json", "r") as file:
            linkedin_data = json.load(file)

        keys = ["full_name", "occupation", "headline", "country_full_name", "city", "state", "experiences", "education", "skills"]
        formatted_linkedin_data = extract_data(linkedin_data, keys)
        formatted_resume_data = extract_key_info(extracted_text)

        with open("formatted_linkedin.json", "w") as file:
            json.dump(formatted_linkedin_data, file, indent=4)

        with open("resume_data.json", 'w') as file:
            json.dump(formatted_resume_data, file, indent=4)

        with open("resume_data.json", "r") as file:
            formatted_resume_data = json.load(file)
        
        with open("formatted_linkedin.json", "r") as file:
            formatted_linkedin_data = json.load(file)
        st.header("Comparison Of Resume and LinkedIn Data")
        result = compare_data(formatted_linkedin_data, formatted_resume_data)
        st.write("Trueness of the provided information: ",result)

# Display common questions based on triggers
st.sidebar.header("Trigger-Based Questions")
display_common_questions()

# Main content area
if st.session_state.current_document:
    st.header(f"Analysis for {st.session_state.current_document}")

    # Display resume analysis
    analysis = st.session_state.resume_analyses[st.session_state.current_document]
    st.markdown(analysis)

    # Display custom trigger results
    display_custom_trigger_results(st.session_state.trigger_faqs.get(st.session_state.current_document, {}))
    
    # Add button to manually reanalyze with all current triggers
    if st.button("Reanalyze CV with Current Triggers"):
        with st.spinner("Reanalyzing CV..."):
            if reanalyze_custom_triggers(st.session_state.current_document):
                st.success("CV reanalyzed successfully!")
            else:
                st.error("Failed to reanalyze CV")

    st.header("Chat with Resume")
    user_question = st.text_input("Ask a question about the resume:")
    if st.button("Ask"):
        with st.spinner("Processing question..."):
            response = chat_with_document(st.session_state.current_document, user_question)
            if response:
                st.session_state.chat_history.append((user_question, response))

    # Display chat history
    if st.session_state.chat_history:
        st.header("Chat History")
        for question, answer in st.session_state.chat_history:
            st.markdown(f"**Q:** {question}")
            st.markdown(f"**A:** {answer}")
            st.markdown("---")

else:
    st.info("Please upload a resume and job description to begin analysis.")



if __name__ == '__main__':
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

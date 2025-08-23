# class BaseAgent:
#     """Base agent class with common functionality"""
    
#     def __init__(self, name, description, vector_db_path=None):
#         self.name = name
#         self.description = description
#         self.vector_db = None
#         self.vector_db_path = vector_db_path
        
#     def initialize(self):
#         """Load the vector database for this agent"""
#         if self.vector_db_path:
#             self.vector_db = self._load_vector_db()
            
#     def _load_vector_db(self):
#         """Load vector database from the specified path"""
#         import os
#         from langchain_community.vectorstores import Chroma
#         from langchain_ollama import OllamaEmbeddings
        
#         if not os.path.exists(self.vector_db_path):
#             return None
            
#         try:
#             return Chroma(
#                 persist_directory=self.vector_db_path,
#                 embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
#             )
#         except Exception as e:
#             print(f"Error loading vector DB for {self.name}: {e}")
#             return None
    
#     def can_handle_query(self, query):
#         """Determine if this agent can handle the given query"""
#         # Base implementation returns False
#         return False
        
#     def retrieve_context(self, query, k=3):
#         """Retrieve relevant context for the query"""
#         if not self.vector_db:
#             return []
            
#         # Use your existing document retrieval logic
#         from chatbot_logic import retrieve_documents
#         return retrieve_documents(self.vector_db, query, k)
        
#     def generate_response(self, query, contexts):
#         """Generate a response based on the query and contexts"""
#         # Base implementation
#         return f"Hello, I'm {self.name}. I don't have specific information on that topic."


# class AdmissionsAgent(BaseAgent):
#     """Agent specialized in handling admissions queries"""
    
#     def __init__(self):
#         super().__init__(
#             name="Admissions Assistant",
#             description="Specialist in admissions processes, requirements, and applications",
#             vector_db_path="./vector_db/admissions"
#         )
        
#     def can_handle_query(self, query):
#         """Check if query is related to admissions"""
#         admissions_keywords = [
#             "apply", "admission", "application", "enroll", "enrollment", 
#             "register", "registration", "transfer", "credit", "transcript",
#             "deadline", "requirement", "qualify", "qualification", "eligibility"
#         ]
        
#         query_lower = query.lower()
#         return any(keyword in query_lower for keyword in admissions_keywords)
        
#     def generate_response(self, query, contexts):
#         """Generate admissions-specific responses"""
#         if not contexts:
#             return "I don't have specific information about that admissions question. Please contact the Division of Admissions directly."
        
#         # Generate response using your existing answer generation logic but with a personality twist
#         import os
#         from openai import AzureOpenAI
        
#         AZURE_API_KEY = os.getenv("AZURE_API_KEY")
#         AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
#         AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
#         AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
        
#         azure_client = AzureOpenAI(
#             api_version=AZURE_API_VERSION,
#             azure_endpoint=AZURE_API_ENDPOINT,
#             api_key=AZURE_API_KEY,
#         )
        
#         combined_context = "\n\n---\n\n".join(contexts)
#         prompt = f"""You are an admissions assistant at a university. Your name is {self.name}.
#         Use the following context to answer the question concisely and helpfully.
        
#         Context:
#         {combined_context}
        
#         Question: {query}
        
#         Respond as a knowledgeable admissions professional. Be helpful but concise.
#         Only answer based on the given context. If you can't find an answer, politely direct
#         the user to contact the Division of Admissions and Credit Evaluation.
#         """
        
#         try:
#             response = azure_client.chat.completions.create(
#                 model=AZURE_DEPLOYMENT_NAME,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful university admissions assistant."},
#                     {"role": "user", "content": prompt}
#                 ]
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return "I'm sorry, I'm having trouble accessing my knowledge base right now. Please try again later or contact the Division of Admissions directly."


# class FinanceAgent(BaseAgent):
#     """Agent specialized in handling finance queries"""
    
#     def __init__(self):
#         super().__init__(
#             name="Finance Advisor",
#             description="Specialist in fees, scholarships, and financial matters",
#             vector_db_path="./vector_db/finance"
#         )
        
#     def can_handle_query(self, query):
#         """Check if query is related to finance"""
#         finance_keywords = [
#             "fee", "fees", "tuition", "scholarship", "financial aid", "payment",
#             "pay", "cost", "expense", "bill", "billing", "refund", "grant", "loan",
#             "funding", "stipend", "budget", "installment", "discount"
#         ]
        
#         query_lower = query.lower()
#         return any(keyword in query_lower for keyword in finance_keywords)
        
#     def generate_response(self, query, contexts):
#         """Generate finance-specific responses"""
#         if not contexts:
#             return "I don't have specific information about that financial question. Please contact the Division of Finance directly."
        
#         # Generate response using Azure OpenAI with finance personality
#         import os
#         from openai import AzureOpenAI
        
#         AZURE_API_KEY = os.getenv("AZURE_API_KEY")
#         AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
#         AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
#         AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
        
#         azure_client = AzureOpenAI(
#             api_version=AZURE_API_VERSION,
#             azure_endpoint=AZURE_API_ENDPOINT,
#             api_key=AZURE_API_KEY,
#         )
        
#         combined_context = "\n\n---\n\n".join(contexts)
#         prompt = f"""You are a finance advisor at a university. Your name is {self.name}.
#         Use the following context to answer the question precisely and accurately.
        
#         Context:
#         {combined_context}
        
#         Question: {query}
        
#         Respond as a precise and detail-oriented finance professional. Mention specific 
#         numbers and dates when available. Only answer based on the given context. If you 
#         can't find an answer, politely direct the user to contact the Division of Finance.
#         """
        
#         try:
#             response = azure_client.chat.completions.create(
#                 model=AZURE_DEPLOYMENT_NAME,
#                 messages=[
#                     {"role": "system", "content": "You are a precise university finance advisor."},
#                     {"role": "user", "content": prompt}
#                 ]
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return "I'm sorry, I'm having trouble accessing my financial database right now. Please try again later or contact the Division of Finance directly."


# class ExaminationAgent(BaseAgent):
#     """Agent specialized in handling examination and course queries"""
    
#     def __init__(self):
#         super().__init__(
#             name="Academic Coordinator",
#             description="Specialist in courses, exams, grades, and academic policies",
#             vector_db_path="./vector_db/examinations"
#         )
        
#     def can_handle_query(self, query):
#         """Check if query is related to examinations and academics"""
#         exam_keywords = [
#             "exam", "examination", "test", "course", "class", "grade", "grading",
#             "syllabus", "curriculum", "schedule", "timetable", "assignment", 
#             "assessment", "credit", "semester", "term", "degree", "diploma",
#             "certificate", "major", "minor", "faculty", "professor", "lecturer"
#         ]
        
#         query_lower = query.lower()
#         return any(keyword in query_lower for keyword in exam_keywords)
        
#     def generate_response(self, query, contexts):
#         """Generate academic-specific responses"""
#         if not contexts:
#             return "I don't have specific information about that academic question. Please contact the Department of Examination and Awards directly."
        
#         # Generate response using Azure OpenAI with academic personality
#         import os
#         from openai import AzureOpenAI
        
#         AZURE_API_KEY = os.getenv("AZURE_API_KEY")
#         AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
#         AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
#         AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
        
#         azure_client = AzureOpenAI(
#             api_version=AZURE_API_VERSION,
#             azure_endpoint=AZURE_API_ENDPOINT,
#             api_key=AZURE_API_KEY,
#         )
        
#         combined_context = "\n\n---\n\n".join(contexts)
#         prompt = f"""You are an academic coordinator at a university. Your name is {self.name}.
#         Use the following context to answer the question clearly and informatively.
        
#         Context:
#         {combined_context}
        
#         Question: {query}
        
#         Respond as a knowledgeable academic professional. Be educational but approachable.
#         Only answer based on the given context. If you can't find an answer, politely direct
#         the user to contact the Department of Examination and Awards.
#         """
        
#         try:
#             response = azure_client.chat.completions.create(
#                 model=AZURE_DEPLOYMENT_NAME,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful university academic coordinator."},
#                     {"role": "user", "content": prompt}
#                 ]
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return "I'm sorry, I'm having trouble accessing my academic database right now. Please try again later or contact the Department of Examination and Awards directly."


# class GeneralAgent(BaseAgent):
#     """General agent for handling queries that don't fit specific departments"""
    
#     def __init__(self):
#         super().__init__(
#             name="University Information Assistant",
#             description="General knowledge about the university"
#         )
        
#     def can_handle_query(self, query):
#         """Default agent that can handle any query"""
#         return True
        
#     def generate_response(self, query, contexts):
#         """Generate general responses"""
#         import os
#         from openai import AzureOpenAI
        
#         AZURE_API_KEY = os.getenv("AZURE_API_KEY")
#         AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
#         AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
#         AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
        
#         azure_client = AzureOpenAI(
#             api_version=AZURE_API_VERSION,
#             azure_endpoint=AZURE_API_ENDPOINT,
#             api_key=AZURE_API_KEY,
#         )
        
#         prompt = f"""You are a general university information assistant. Your name is {self.name}.
        
#         Question: {query}
        
#         Respond as a helpful university assistant. For this query, I don't have specific information,
#         so I'll provide a general response and suggest which department might help.
#         """
        
#         try:
#             response = azure_client.chat.completions.create(
#                 model=AZURE_DEPLOYMENT_NAME,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful university information assistant."},
#                     {"role": "user", "content": prompt}
#                 ]
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return "I'm sorry, I don't have specific information about that. Please contact the university's main information desk for assistance."
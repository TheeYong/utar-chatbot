# import json

# class CollaborativeSystem:
#     """Enables multiple agents to collaborate on complex queries"""
    
#     def __init__(self, agent_manager):
#         self.agent_manager = agent_manager
        
#     def handle_complex_query(self, query, session_id=None):
#         """Handle complex queries that might require multiple agents"""
#         # Analyze the query to determine if it spans multiple domains
#         domains = self._identify_domains(query)
        
#         if len(domains) <= 1:
#             # Simple query - use standard routing
#             return self.agent_manager.process_query(query, session_id)
        
#         # Complex query spanning multiple domains
#         responses = []
#         consolidated_contexts = []
        
#         # Collect responses and contexts from all relevant agents
#         for domain in domains:
#             for agent in self.agent_manager.agents:
#                 if agent.name.lower().startswith(domain.lower()):
#                     contexts = agent.retrieve_context(query)
#                     consolidated_contexts.extend(contexts)
                    
#                     # Get preliminary response from this agent
#                     response = agent.generate_response(query, contexts)
#                     responses.append(f"From {agent.name}: {response}")
        
#         # Generate a synthesized response using the primary agent
#         primary_agent = self.agent_manager.get_agent_for_query(query)
        
#         # Use Azure OpenAI to synthesize a coherent response from multiple perspectives
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
        
#         # Construct a prompt that combines all agent perspectives
#         combined_responses = "\n\n".join(responses)
#         prompt = f"""You are a university chatbot coordinator. The following question spans multiple departments:

# Question: {query}

# You have received these individual responses from different department specialists:

# {combined_responses}

# Please synthesize a comprehensive, coherent response that addresses all aspects of the question while
# avoiding repetition and contradictions. Acknowledge that this question required input from multiple departments.
# """
        
#         try:
#             response = azure_client.chat.completions.create(
#                 model=AZURE_DEPLOYMENT_NAME,
#                 messages=[
#                     {"role": "system", "content": "You are a university chatbot coordinator."},
#                     {"role": "user", "content": prompt}
#                 ]
#             )
            
#             synthesized_response = response.choices[0].message.content.strip()
            
#             # Add to conversation memory if session_id is provided
#             if session_id and hasattr(primary_agent, 'memory'):
#                 primary_agent.memory.add_message(session_id, "user", query)
#                 primary_agent.memory.add_message(session_id, "assistant", synthesized_response)
                
#             return {
#                 "agent_name": "Multi-Department Coordinator",
#                 "agent_description": "Synthesizes information from multiple university departments",
#                 "response": synthesized_response,
#                 "contributing_departments": domains
#             }
            
#         except Exception as e:
#             print(f"Error generating collaborative response: {e}")
#             return self.agent_manager.process_query(query, session_id)
    
#     def _identify_domains(self, query):
#         """Identify which domains a query belongs to"""
#         # Use Azure OpenAI to classify the query into multiple domains
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
        
#         prompt = f"""Analyze this query and determine which university departments it relates to:

# Query: "{query}"

# Choose from these departments:
# - Admissions: admissions, applications, enrollment, registration
# - Finance: fees, payments, scholarships, financial aid
# - Examinations: courses, exams, grades, academic policies

# Return a JSON array of department names that this query relates to. For example:
# ["Admissions", "Finance"]

# If the query only relates to one department, return just that one. If it's general or unclear, return ["General"].
# """
        
#         try:
#             response = azure_client.chat.completions.create(
#                 model=AZURE_DEPLOYMENT_NAME,
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"}
#             )
            
#             content = response.choices[0].message.content
#             result = json.loads(content)
            
#             if "departments" in result:
#                 return result["departments"]
#             else:
#                 # Try to extract a JSON array from the text
#                 import re
#                 json_match = re.search(r'\[.*?\]', content)
#                 if json_match:
#                     return json.loads(json_match.group())
#                 else:
#                     return ["General"]
                    
#         except Exception as e:
#             print(f"Error in domain identification: {e}")
#             return ["General"]


# # Update the agent manager to include collaborative capabilities
# class AgentManager:
#     """Manages multiple specialized agents and routes queries to the appropriate one"""
    
#     def __init__(self):
#         self.agents = []
#         self.initialize_agents()
#         self.collaborative_system = CollaborativeSystem(self)
        
#     # ... (rest of the class code)
    
#     def process_query(self, query, session_id=None):
#         """Process a user query with the option for collaborative problem solving"""
#         # Determine query complexity
#         is_complex = self._is_complex_query(query)
        
#         if is_complex:
#             # Use collaborative system for complex queries
#             return self.collaborative_system.handle_complex_query(query, session_id)
        
#         # Simple query - use standard routing
#         agent = self.get_agent_for_query(query)
#         print(f"Selected agent: {agent.name}")
        
#         # Get context from the agent's knowledge base
#         contexts = agent.retrieve_context(query)
        
#         # Generate and return the response
#         response = agent.generate_response(query, contexts, session_id)
        
#         return {
#             "agent_name": agent.name,
#             "agent_description": agent.description,
#             "response": response
#         }
        
#     def _is_complex_query(self, query):
#         """Determine if a query is complex and might need multiple agents"""
#         # A basic heuristic - check the number of distinct topics
#         query_lower = query.lower()
        
#         # Define keywords for different domains
#         admissions_keywords = ["admission", "apply", "application", "register"]
#         finance_keywords = ["fee", "payment", "scholarship", "financial"]  
#         academic_keywords = ["course", "exam", "grade", "class"]
        
#         # Count how many domains are mentioned
#         domains_mentioned = 0
#         if any(keyword in query_lower for keyword in admissions_keywords):
#             domains_mentioned += 1
#         if any(keyword in query_lower for keyword in finance_keywords):
#             domains_mentioned += 1
#         if any(keyword in query_lower for keyword in academic_keywords):
#             domains_mentioned += 1
            
#         # Consider it complex if it mentions at least 2 domains
#         return domains_mentioned >= 2
class ConversationMemory:
    """Maintains conversation history for stateful interactions"""
    
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.conversations = {}  # Dictionary to store conversations by session_id
        
    # def add_message(self, session_id, role, content, agent_id=None):
    #     """Add a message to the conversation history"""
    #     # Create conversation for this session if it doesn't exist
    #     if session_id not in self.conversations:
    #         self.conversations[session_id] = []
            
    #     # Add the message
    #     self.conversations[session_id].append({
    #         "role": role,
    #         "content": content,
    #         "agent_id": agent_id
    #     })
        
    #     # Limit history size
    #     if len(self.conversations[session_id]) > self.max_history * 2:  # *2 because we store pairs of messages
    #         # Remove oldest messages to maintain max_history
    #         self.conversations[session_id] = self.conversations[session_id][-self.max_history*2:]
            
    def add_message(self, session_id, role, content, agent_id=None):
        """Add a message to the conversation history"""
        # Create conversation for this session if it doesn't exist
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            
        # Add the message as a dictionary
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "agent_id": agent_id
        })
        
        # Limit history size
        if len(self.conversations[session_id]) > self.max_history * 2:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history*2:]

    def get_conversation(self, session_id):
        """Get the conversation history for a session"""
        return self.conversations.get(session_id, [])
        
    def clear_conversation(self, session_id):
        """Clear the conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
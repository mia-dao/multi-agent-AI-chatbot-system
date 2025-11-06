"""
Response Generator Agent - Creates final responses
"""
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.state import AgentState


class ResponseGeneratorAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful, friendly AI assistant.
            Generate clear, concise, and helpful responses.
            
            Guidelines:
            - Be conversational and natural
            - Provide actionable information
            - Be empathetic and understanding
            - Keep responses concise but complete
            - Use formatting for clarity when needed"""),
            ("user", """
            User Query: {input}
            Intent: {intent}
            Retrieved Information: {retrieved_info}
            
            Generate a helpful response:
            """)
        ])
    
    def process(self, state: AgentState) -> Dict:
        """Generate response"""
        chain = self.prompt | self.llm
        result = chain.invoke({
            "input": state["user_input"],
            "intent": state.get("intent", "unknown"),
            "retrieved_info": state.get("retrieved_info", "")
        })
        
        return {
            "response": result.content
        }
"""
Intent Classification Agent - Analyzes user input to determine intent
"""
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.state import AgentState


class IntentClassifierAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert intent classifier for a chatbot system.
            Analyze the user's message and classify it into one of these categories:
            - question: User is asking a question
            - information: User wants specific information
            - help: User needs assistance or support
            - greeting: User is greeting or making small talk
            - feedback: User is providing feedback
            - task: User wants to perform an action
            
            Respond with JSON: {{"intent": "category", "confidence": 0.0-1.0, "context": "brief context"}}"""),
            ("user", "{input}")
        ])
    
    def process(self, state: AgentState) -> Dict:
        """Classify the user's intent"""
        chain = self.prompt | self.llm
        result = chain.invoke({"input": state["user_input"]})
        
        # Parse the response
        import json
        try:
            parsed = json.loads(result.content)
            return {
                "intent": parsed.get("intent", "unknown"),
                "confidence": parsed.get("confidence", 0.5),
                "context": {"intent_context": parsed.get("context", "")}
            }
        except json.JSONDecodeError:
            return {
                "intent": "unknown",
                "confidence": 0.3,
                "context": {"intent_context": "Failed to parse"}
            }
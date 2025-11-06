"""
LangGraph Workflow - Orchestrates multi-agent system
"""
from langgraph.graph import StateGraph, END
from models.state import AgentState
from agents.intent_classifier import IntentClassifierAgent
from agents.information_retriever import InformationRetrieverAgent
from agents.response_generator import ResponseGeneratorAgent
from agents.quality_checker import QualityCheckerAgent


class ChatbotWorkflow:
    def __init__(self):
        self.intent_classifier = IntentClassifierAgent()
        self.info_retriever = InformationRetrieverAgent()
        self.response_generator = ResponseGeneratorAgent()
        self.quality_checker = QualityCheckerAgent()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("retrieve_info", self._retrieve_info_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("check_quality", self._check_quality_node)
        
        # Define edges
        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "retrieve_info")
        workflow.add_edge("retrieve_info", "generate_response")
        workflow.add_edge("generate_response", "check_quality")
        
        # Conditional edge based on quality
        workflow.add_conditional_edges(
            "check_quality",
            self._should_refine,
            {
                "refine": "generate_response",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _classify_intent_node(self, state: AgentState):
        """Node for intent classification"""
        result = self.intent_classifier.process(state)
        return result
    
    def _retrieve_info_node(self, state: AgentState):
        """Node for information retrieval"""
        result = self.info_retriever.process(state)
        return result
    
    def _generate_response_node(self, state: AgentState):
        """Node for response generation"""
        result = self.response_generator.process(state)
        return result
    
    def _check_quality_node(self, state: AgentState):
        """Node for quality checking"""
        result = self.quality_checker.process(state)
        return result
    
    def _should_refine(self, state: AgentState) -> str:
        """Determine if response needs refinement"""
        # Limit iterations to prevent loops
        if state.get("iteration_count", 0) >= 2:
            return "end"
        
        needs_refinement = state.get("needs_refinement", False)
        quality_score = state.get("quality_score", 1.0)
        
        if needs_refinement and quality_score < 0.7:
            return "refine"
        return "end"
    
    def process(self, user_input: str):
        """Process user input through the workflow"""
        initial_state = {
            "messages": [],
            "user_input": user_input,
            "intent": None,
            "confidence": None,
            "retrieved_info": None,
            "response": None,
            "quality_score": None,
            "needs_refinement": False,
            "iteration_count": 0,
            "context": {}
        }
        
        result = self.graph.invoke(initial_state)
        return result
import re
from typing import Dict, Any

def extract_context(conversation_text: str) -> Dict[str, Any]:
    text_lower = conversation_text.lower()
    
    context = {
        "roles": [],
        "skills": [],
        "seniority": [],
        "duration_preference": None,
        "remote_preference": None,
        "adaptive_preference": None,
        "test_type_preferences": [],
        "comparisons": []
    }
    
    # Simple extraction logic based on keywords
    roles = [
        "developer", "software engineer", "java developer", "backend developer", 
        "frontend developer", "full stack developer", "data analyst", "data scientist", 
        "business analyst", "sales representative", "sales manager", "customer service", 
        "support", "contact center", "manager", "graduate", "intern", "administrator", 
        "finance", "accountant", "marketing", "leadership", "supervisor", "engineer",
        "sales", "analyst", "designer", "consultant"
    ]
    for role in roles:
        if role in text_lower:
            context["roles"].append(role)
            
    skills = [
        "java", "core java", "java 8", "python", "sql", "c++", "c#", ".net", 
        "javascript", "react", "angular", "node", "html", "css", "excel", "aws", 
        "cloud", "devops", "docker", "kubernetes", "data science", "machine learning", 
        "analytics", "communication", "stakeholder", "customer service", "sales", 
        "leadership", "reasoning", "numerical", "verbal", "cognitive", "personality", 
        "opq", "gsa", "verify"
    ]
    for skill in skills:
        if skill in text_lower:
            context["skills"].append(skill)
            
    seniorities = {
        "fresher": "Entry-Level",
        "entry": "Entry-Level",
        "entry-level": "Entry-Level",
        "graduate": "Graduate",
        "mid": "Mid-Professional",
        "mid-level": "Mid-Professional",
        "3 years": "Mid-Professional",
        "4 years": "Mid-Professional",
        "5 years": "Mid-Professional",
        "senior": "Professional Individual Contributor", # or Manager, handled loosely
        "lead": "Manager",
        "manager": "Manager",
        "supervisor": "Supervisor",
        "director": "Director",
        "executive": "Executive"
    }
    for kw, val in seniorities.items():
        if kw in text_lower:
            if val not in context["seniority"]:
                context["seniority"].append(val)
                
    if re.search(r"short|quick|under 20|fast", text_lower):
        context["duration_preference"] = "short"
        
    if "remote" in text_lower or "online" in text_lower:
        context["remote_preference"] = "yes"
        
    if "adaptive" in text_lower:
        context["adaptive_preference"] = "yes"
        
    test_types = {
        "programming": "Knowledge & Skills",
        "coding": "Knowledge & Skills",
        "technical": "Knowledge & Skills",
        "reasoning": "Ability & Aptitude",
        "numerical": "Ability & Aptitude",
        "verbal": "Ability & Aptitude",
        "cognitive": "Ability & Aptitude",
        "aptitude": "Ability & Aptitude",
        "personality": "Personality & Behavior",
        "behavior": "Personality & Behavior",
        "behaviour": "Personality & Behavior",
        "motivation": "Personality & Behavior",
        "leadership": "Competencies",
        "teamwork": "Competencies",
        "competency": "Competencies",
        "situational": "Biodata & Situational Judgment",
        "judgment": "Biodata & Situational Judgment",
        "judgement": "Biodata & Situational Judgment"
    }
    for kw, val in test_types.items():
        if kw in text_lower:
            if val not in context["test_type_preferences"]:
                context["test_type_preferences"].append(val)

    # Check for specific known assessment names if comparing
    # e.g. "OPQ", "GSA", "Verify G"
    compare_targets = ["opq", "gsa", "verify g", "java 8", "core java"]
    for target in compare_targets:
        if target in text_lower:
            context["comparisons"].append(target)
            
    return context

def has_enough_context(context: Dict[str, Any], conversation_text: str) -> bool:
    if context["roles"] or context["skills"] or context["comparisons"]:
        return True
    
    # If the user gives a long description (e.g. pasted JD)
    if len(conversation_text.split()) > 20:
        return True
        
    return False

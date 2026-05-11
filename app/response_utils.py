from typing import List, Dict, Any
from app.schemas import ChatResponse, Recommendation
from app.catalog_loader import get_catalog


def build_deterministic_comparison_reply(items: List[Dict[str, Any]]) -> str:
    """
    Build a grounded comparison reply from catalog item fields only.
    Does not use LLM. Does not hardcode any assessment names.
    """
    if not items:
        return "I could not find enough matching SHL catalog entries to compare those assessments."

    parts = []
    for item in items:
        name = item.get("name", "Unknown")
        test_type = item.get("test_type", "")
        keys = ", ".join(item.get("keys", []))
        duration = item.get("duration", "")
        job_levels = ", ".join(item.get("job_levels", []))
        description = item.get("description", "")

        entry = f"**{name}**"
        if test_type:
            entry += f" (Type: {test_type})"
        if keys:
            entry += f": Categorized under {keys}."
        if duration:
            entry += f" Duration: {duration}."
        if job_levels:
            entry += f" Suitable for: {job_levels}."
        if description:
            # Truncate long descriptions
            short_desc = description[:200] + "..." if len(description) > 200 else description
            entry += f" {short_desc}"
        parts.append(entry)

    intro = "Based on the SHL catalog, here is a comparison of the requested assessments:\n\n"
    return intro + "\n\n".join(parts)

def validate_and_format_response(reply: str, items: List[Dict[str, Any]], intent: str, end_of_conversation: bool) -> dict:
    _, valid_urls = get_catalog()
    
    recommendations = []
    
    # Only return recommendations if we actually meant to recommend or refine and we have items
    if intent in ["recommend", "refine"] and items:
        # Pre-filter items to detect JFAs
        def is_jfa(i: dict) -> bool:
            text = f"{i.get('name', '')} {i.get('url', '')}".lower()
            return any(t in text for t in [" solution", "-solution", "job-focused-assessment", "job focused assessment", "jfa"])
            
        non_jfa_items = [i for i in items if not is_jfa(i)]
        has_valid_non_jfa = len(non_jfa_items) > 0
        
        # Take up to 10
        for item in items:
            if len(recommendations) >= 10:
                break
                
            if has_valid_non_jfa and is_jfa(item):
                continue
                
            if item["url"] in valid_urls:
                rec = Recommendation(
                    name=item["name"],
                    url=item["url"],
                    test_type=item["test_type"]
                )
                recommendations.append(rec)
                    
        # Fallback if empty
        if not recommendations:
            for item in items:
                if len(recommendations) >= 10:
                    break
                if item["url"] in valid_urls:
                    rec = Recommendation(
                        name=item["name"],
                        url=item["url"],
                        test_type=item["test_type"]
                    )
                    recommendations.append(rec)
                
    # If the LLM failed or generated no reply, ensure a string exists
    if not reply:
        reply = "Here is the requested information."
        
    response = ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation
    )
    
    # Return as dict for FastAPI to serialize
    return response.model_dump()

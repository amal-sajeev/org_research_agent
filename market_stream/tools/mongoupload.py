import os
from pymongo import MongoClient
import requests,json

def create_blank_project(project_id: str):
    from pymongo import MongoClient
    connection_string = os.getenv("MONGO_DB_CONNECTOR")
    client = MongoClient(connection_string)
    db = client["sales_reports"]
    collection = db["projects"]

    # Check if a document with the same project_id exists
    existing_doc = collection.find_one({"project_id": project_id})
    
    if existing_doc:
        print(f"Document with project_id '{project_id}' already exists.")
        return True  # No document is created, but returning False would cause pointless tool reruns 

    # If not found, insert the blank document
    document = {
        "project_id": project_id,
        "client_org_research": "",
        "prospect_research": "",
        "market_segment": "",
        "target_org_research": "",
        "client_org_research_html": "",
        "market_context_html": "",
        "prospect_research_html": "",
        "market_segment_html": "",
        "target_org_research_html": ""
    }

    collection.insert_one(document)
    print(f"New document created for project_id '{project_id}'.")
    return True 

def update_project_report(project_id: str, report: str, report_type: str, html_report: str = ""):
    """
    Updates the report field (report_type) in the project document with the given project_id.
    """
    allowed_fields = {
        "client_org_research",
        "market_context",
        "prospect_research",
        "market_segment",
        "target_org_research",
        "client_org_research_html",
        "market_context_html",
        "prospect_research_html",
        "market_segment_html",
        "target_org_research_html"
    }

    if report_type not in allowed_fields:
        raise ValueError(f"Invalid report_type. Must be one of: {', '.join(allowed_fields)}")

    mongo_uri = os.getenv("MONGO_DB_CONNECTOR")
    if not mongo_uri:
        raise ValueError("MONGO_DB_CONNECTOR environment variable is not set.")

    client = MongoClient(mongo_uri)
    db = client["sales_reports"]
    collection = db["projects"]

    result = collection.update_one(
        {"project_id": project_id},
        {"$set": {report_type: report, f"{report_type}_html":html_report}}
    )

    if html_report !=None or html_report == "":
        format = "agent_status"
    else:
        format = "sub_status"

    if report_type == "prospect_research":
        message = "Completed"
    else:
        message = "updated"

    requests.put(f"https://stu.globalknowledgetech.com:8444/project/project-status-update/{project_id}/",headers = {'Content-Type': 'application/json'}, data = json.dumps({format: f"{report_type} {message}",}))

    client.close()

    if result.matched_count == 0:
        raise ValueError(f"No project found with project_id '{project_id}'")
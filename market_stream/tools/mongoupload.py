import os
from pymongo import MongoClient
import requests,json
from google.adk.agents.callback_context import CallbackContext

def announce_markdown_upload(project_id: str, report_type: str):
    """Store prospect research report after prospect_researcher completes"""
    try:
        project_id = project_id.replace('"','')

        library = {
        "market_context": "market_intelligence_agent",
        "market_segment": "segmentation_intelligence_agent",
        "target_org_research": "sales_intelligence_agent",
        "prospect_research": "prospect_researcher"
        }
        requests.put(f"https://stu.globalknowledgetech.com:8444/project/project-status-update/{project_id}/",headers = {'Content-Type': 'application/json'}, data = json.dumps({"sub_status": f"{report_type} updated"}))
        
        
        print("###################################################################################")
        print("###################################################################################")
        print(f"{report_type} report stored successfully for project {project_id}")
        print(f'@sub_status {report_type} updated for project {project_id}')
        print("###################################################################################")
        print("###################################################################################")
        
    except Exception as e:
        print(f"Error announcing markdown report storage: {e}")
        

def announce_markdown_finish(callback_context:CallbackContext):
    project_id = callback_context.state["project_id"]
    try:
        project_id = project_id.replace('"','')
        requests.put(f"https://stu.globalknowledgetech.com:8444/project/project-status-update/{project_id}/",headers = {'Content-Type': 'application/json'}, data = json.dumps({"sub_status": f"Completed"}))
        print('@{"sub_status": f"Completed"}')
    except Exception as e:
        print(f"Error announcing markdown completion: {e}")

def announce_html_upload(project_id: str, report_type: str):
    """Store prospect research report after prospect_researcher completes"""
    try:
        project_id = project_id.replace('"','')

        library = {
        "market_context": "context_html",
        "market_segment": "seg_html",
        "target_org_research": "target_html"
        }
        
        requests.put(f"https://stu.globalknowledgetech.com:8444/project/project-status-update/{project_id}/",headers = {'Content-Type': 'application/json'}, data = json.dumps({"agent_status": f"{report_type} updated"}))

        print("###################################################################################")
        print("###################################################################################")
        print(f"{report_type} report stored successfully for project {project_id}")
        print(f'@agent_status {report_type} updated for project {project_id}')
        print("###################################################################################")
        print("###################################################################################")
        
    except Exception as e:
        print(f"Error storing prospect research report: {e}")

def announce_html_finish(callback_context:CallbackContext):
    project_id = callback_context.state["project_id"]
    try:
        project_id = project_id.replace('"','')
        requests.put(f"https://stu.globalknowledgetech.com:8444/project/project-status-update/{project_id}/",headers = {'Content-Type': 'application/json'}, data = json.dumps({"agent_status": f"Completed"}))
        print({"agent_status": f"Completed"})
    except Exception as e:
        print(f"Error announcing markdown completion: {e}")

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

    client.close()
    if html_report != None or html_report != "":
        announce_markdown_upload(project_id,report_type)
    else:
        announce_html_upload(project_id,report_type)

    if result.matched_count == 0:
        raise ValueError(f"No project found with project_id '{project_id}'")
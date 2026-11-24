import os
import re
import sys
import requests

JIRA_BASE_URL = os.environ["JIRA_BASE_URL"]
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]

PR_TITLE = os.environ.get("PR_TITLE", "")
BRANCH_NAME = os.environ.get("BRANCH_NAME", "")


def extract_ticket_key(text):
    match = re.search(
        r"\b[A-Z][A-Z0-9]+-\d+\b", text
    )
    return match.group(0) if match else None


def fetch_jira_ticket(ticket_key):
    url = (
        f"{JIRA_BASE_URL.rstrip('/')}"
        f"/rest/api/3/issue/{ticket_key}"
    )

    response = requests.get(
        url,
        auth=(JIRA_EMAIL, JIRA_API_TOKEN),
        headers={"Accept": "application/json"},
        params={
            "fields": "summary,description,status"
        },
        timeout=30
    )

    if response.status_code == 404:
        raise Exception(
            f"Ticket {ticket_key} not found or inaccessible."
        )

    response.raise_for_status()

    return response.json()


def main():
    title_key = extract_ticket_key(PR_TITLE)
    branch_key = extract_ticket_key(BRANCH_NAME)

    if title_key and branch_key and title_key != branch_key:
        print("PR title and branch have different Jira keys.")
        sys.exit(1)

    ticket_key = title_key or branch_key

    if not ticket_key:
        print("No Jira ticket key found.")
        sys.exit(1)

    # Change SAMPLE to your actual Jira space key.
    if not ticket_key.startswith("KAN-"):
        print("Ticket belongs to an unexpected Jira space.")
        sys.exit(1)

    print(f"Validating Jira ticket: {ticket_key}")

    try:
        ticket = fetch_jira_ticket(ticket_key)

        fields = ticket["fields"]

        print("Jira ticket successfully validated.")
        print("Summary:", fields["summary"])
        print("Status:", fields["status"]["name"])

        if not fields.get("description"):
            print("Ticket description is missing.")
            sys.exit(1)

        print("Jira validation passed.")

    except Exception as error:
        print(f"Jira validation failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()

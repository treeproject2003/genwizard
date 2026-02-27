```markdown
# Accounts Payable Multi-Agent Automation System

## 1. Project Overview

This project implements a multi-agent, Python-based automation system for end-to-end Accounts Payable (AP) processing. It streamlines invoice intake, validation, ERP entry, payment scheduling, and record maintenance with a clear separation of automation and human-in-the-loop approvals/exceptions. The system is orchestrated by a Supervisor Agent, who delegates and oversees AP workflow stages, ensuring compliance, auditability, and efficient human/agent collaboration.

---

## 2. Architecture Overview

The system is structured as a hierarchical agent network:

- **Supervisor Agent**: Central orchestrator responsible for overall process control, exception handling, and approvals.
- **Utility Agents (AP01)**: Specialized agents that automate specific AP workflow segments (e.g., invoice logging, validation, ERP entry, payment scheduling, record maintenance).
- **Tools**: Modular Python functions assigned to utility agents to perform core AP process logic (e.g., invoice extraction, validation, ERP integration).

Workflow is sequential and event-driven, with branching logic to support human intervention for exceptions and approvals.

---

## 3. Agents & Tools Description

### **Finance Operations Supervisor (Supervisor Agent)**

- **Description**: Supervises the accounts payable process, delegates routine automation to utility agents, intervenes for exception handling and payment approvals, and ensures compliance with financial policies.

#### **Supervised Utility Agent: Generic Accounts Payable Agent AP01**

- **Description**: Automates the end-to-end AP process—receiving and logging invoices, validating invoice details, entering invoices into the ERP, scheduling payments, and maintaining AP records. Collaborates with the supervisor for exception handling and payment batch approval.

  - **Assigned Tools:**

    - **receive_and_log_invoices**
      - *Description*: Receives a list of invoice files, extracts key invoice data fields, validates required fields, and logs structured invoice data for further processing in the accounts payable workflow.
      - *When to use*: When new invoices are received via email or upload and need to be logged for processing.

    - **validate_invoice_details**
      - *Description*: Validates extracted invoice details by cross-checking with purchase orders (PO) and goods receipts (GR). Flags discrepancies for human review and marks invoices as validated if all checks pass.
      - *When to use*: After invoice data extraction, before entry into the accounting system.

    - **enter_invoices_to_erp**
      - *Description*: Enters validated invoices into the ERP system, assigns accounting codes, and logs confirmation details.
      - *When to use*: After invoice validation is complete.

    - **schedule_payments_for_approved_invoices**
      - *Description*: Groups approved invoices into payment batches, generates payment batch file names, and logs details for audit purposes.
      - *When to use*: After invoices are approved for payment.

    - **maintain_ap_records**
      - *Description*: Archives processed invoice data and payment confirmations, organizes records by period and vendor, and generates periodic AP aging report files.
      - *When to use*: After payment processing and at regular reporting intervals.

---

## 4. Workflow Execution Flow

1. **Invoice Receipt & Logging**
   - Supervisor delegates to AP01 to receive/upload and log incoming invoices using `receive_and_log_invoices`.
2. **Invoice Validation**
   - AP01 validates invoices using `validate_invoice_details` (cross-checks with PO/GR).
   - If discrepancies exist, invoices are flagged for supervisor review.
3. **Human Exception Handling**
   - Supervisor reviews flagged items, resolves issues, and updates status.
4. **ERP Entry**
   - AP01 enters validated invoices into ERP using `enter_invoices_to_erp`.
5. **Payment Scheduling**
   - AP01 schedules payments for approved invoices using `schedule_payments_for_approved_invoices`.
   - Payment batches are routed to the supervisor for approval.
6. **Supervisor Payment Approval**
   - Supervisor reviews and approves payment batches.
7. **Record Maintenance & Reporting**
   - AP01 archives records and generates reports using `maintain_ap_records`.
8. **Branching Logic**
   - Human intervention is required for flagged discrepancies and payment batch approvals. Rejected items are routed back to AP01 for correction.

---

## 5. Folder Structure

```
accounts-payable-system/
│
├── supervisor_agent/
│   └── finance_operations_supervisor.py
│
├── ap_agents/
│   ├── invoiceintakeandloggingagent.py
│   ├── invoicevalidationagent.py
│   ├── invoiceentryagent.py
│   ├── paymentschedulingagent.py
│   └── aprecordmaintenanceagent.py
│
├── tools/
│   ├── receive_and_log_invoices_tool.py
│   ├── validate_invoice_details_tool.py
│   ├── enter_invoices_to_erp_tool.py
│   ├── schedule_payments_for_approved_invoices_tool.py
│   └── maintain_ap_records_tool.py
│
├── instructions.txt
├── mcp_config.json
├── .env
└── README.md
```

---

## 6. How to Run

1. **Setup Environment**
   - Install dependencies: `pip install -r requirements.txt`
   - Configure `.env` and `mcp_config.json` as per your environment and API keys.
2. **Start Each Agent**
   - Launch each agent script individually (e.g., `python ap_agents/invoiceintakeandloggingagent.py [PORT]`)
   - Launch the supervisor agent (e.g., `python supervisor_agent/finance_operations_supervisor.py [PORT]`)
3. **Interaction**
   - Use API endpoints exposed by supervisor and utility agents, or connect via the provided UI (if implemented).
   - The supervisor agent orchestrates the workflow, delegating tasks to AP01 as per `instructions.txt`.
4. **Monitoring**
   - All agent actions, exceptions, and workflow transitions are logged for audit and compliance.

---

## 7. Notes / Constraints

- **Human-in-the-loop**: Supervisor agent must provide human input for flagged exceptions and payment batch approvals.
- **Do not bypass supervisor**: All routine automation must follow delegation protocol; exceptions and approvals require explicit supervisor action.
- **Tool Usage**: Tool descriptions and logic are strictly as provided—do not alter or replace core tool logic.
- **Agent/Tool Integrity**: Do not invent or modify agents or tools not defined in the provided scripts and requirements.
- **File Paths & Data**: Always use absolute file paths as required by agents for correct data handoff and retrieval.
- **Compliance**: System supports audit trails and reporting in line with financial best practices and policies.
- **Extensibility**: Further AP logic or agents should follow the established hierarchical delegation and tool assignment pattern.

---
```
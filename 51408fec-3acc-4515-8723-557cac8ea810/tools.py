
from typing import Dict, Any, List

def receive_and_log_invoices_tool(invoice_files: List[str]) -> Dict[str, Any]:
    """
    Description:
        Receives a list of invoice files, extracts key invoice data fields, validates required fields, 
        and logs structured invoice data for further processing in the accounts payable workflow.

    Logic:
        1. Accepts a list of file paths or identifiers to invoice files (PDF, image, or electronic formats).
        2. For each file, extract key invoice fields:
            - vendor
            - date
            - amount
            - invoice number
            - line items (as a list of dicts)
        3. Validate that all required fields are present and in the expected format for each extracted invoice.
        4. If any invoice is missing required fields or has invalid formats, exclude it from the result and log an error (included in output).
        5. Prepare a list of successfully extracted and validated invoice data dicts.
        6. Return the structured invoice data (list of dict) for downstream processing.

    Inputs:
        invoice_files: List[str]
            List of file paths or storage identifiers for the invoice documents.

    Outputs:
        {
            "invoice_data": List[Dict[str, Any]]
                List of dicts, each representing a single invoice with extracted and validated fields.
                Example:
                [
                    {
                        "vendor": "ABC Corp",
                        "date": "2024-06-20",
                        "amount": 1000.0,
                        "invoice_number": "INV-123",
                        "line_items": [
                            {"description": "Item1", "quantity": 2, "unit_price": 500.0}
                        ]
                    },
                    ...
                ]
            "errors": List[Dict[str, Any]]
                List of error dicts for files that could not be processed. Example:
                [
                    {"file": "invoice2.pdf", "error": "Missing invoice number"},
                    ...
                ]
        }

    When to use:
        When new invoices are received via email or upload and need to be logged for processing.
    """
    import re
    from datetime import datetime

    invoice_data = []
    errors = []

    # Dummy invoice parsing logic (since no OCR or real parsing)
    required_fields = ["vendor", "date", "amount", "invoice_number", "line_items"]

    for file_name in invoice_files:
        # Simulated extraction logic: file name contains invoice info (for deterministic demonstration)
        # In real deployment, replace with actual parsing (OCR, PDF, etc.)
        try:
            # Example: invoice-ABC_Corp-20240620-1000.0-INV123.txt
            match = re.match(r".*invoice-([^-\s]+)-(\d{8})-([\d\.]+)-([A-Za-z0-9]+)\.(pdf|txt|jpg|jpeg|png)$", file_name)
            if not match:
                raise ValueError("Filename does not match expected invoice pattern")
            vendor, date_str, amount_str, invoice_num, _ = match.groups()
            try:
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                date_val = date_obj.strftime("%Y-%m-%d")
            except Exception:
                raise ValueError("Invalid date format")
            try:
                amount_val = float(amount_str)
            except Exception:
                raise ValueError("Invalid amount format")
            line_items = [
                {"description": "Generic Item", "quantity": 1, "unit_price": amount_val}
            ]
            invoice_dict = {
                "vendor": vendor,
                "date": date_val,
                "amount": amount_val,
                "invoice_number": invoice_num,
                "line_items": line_items
            }
            # Validation: all required fields present and non-empty
            missing = [f for f in required_fields if not invoice_dict.get(f)]
            if missing:
                raise ValueError(f"Missing fields: {','.join(missing)}")
            invoice_data.append(invoice_dict)
        except Exception as e:
            errors.append({"file": file_name, "error": str(e)})

    return {
        "invoice_data": invoice_data,
        "errors": errors
    }


def validate_invoice_details_tool(
    invoice_data: List[Dict[str, Any]],
    po_data: List[Dict[str, Any]],
    gr_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Description:
        Validates extracted invoice details by cross-checking with purchase orders (PO) and goods receipts (GR).
        Flags discrepancies for human review and marks invoices as validated if all checks pass.

    Logic:
        1. For each invoice:
            a. Attempt to find a matching PO and GR (using invoice_number, PO number, or vendor as keys).
            b. Cross-check amount, vendor, and line items between invoice, PO, and GR.
            c. If any discrepancy is found (mismatched amount, missing PO, missing GR, etc.), 
               add a 'flagged' entry with discrepancy info.
            d. If all checks pass, add to validated list with a 'validated' status.
        2. Output the combined result as list of dicts: 
            - Each invoice dict includes a status key: 'validated' or 'flagged'
            - If flagged, include a 'discrepancies' field listing issues.

    Inputs:
        invoice_data: List[Dict[str, Any]]
            Extracted invoice data (from previous tool).
        po_data: List[Dict[str, Any]]
            List of purchase order dicts.
        gr_data: List[Dict[str, Any]]
            List of goods receipt dicts.

    Outputs:
        {
            "validated_invoices": List[Dict[str, Any]]
                Each dict includes:
                    - all invoice fields
                    - 'status': "validated" or "flagged"
                    - 'discrepancies': List[str] (if flagged)
        }

    When to use:
        After invoice data extraction, before entry into the accounting system.
    """
    validated_invoices = []

    # Index PO and GR by PO number and invoice number for quick lookup
    po_index = {}
    for po in po_data:
        po_num = po.get("po_number") or po.get("invoice_number")
        if po_num:
            po_index.setdefault(po_num, []).append(po)

    gr_index = {}
    for gr in gr_data:
        gr_num = gr.get("gr_number") or gr.get("invoice_number") or gr.get("po_number")
        if gr_num:
            gr_index.setdefault(gr_num, []).append(gr)

    for inv in invoice_data:
        status = "validated"
        discrepancies = []
        invoice_number = inv.get("invoice_number")
        vendor = inv.get("vendor")
        amount = inv.get("amount")
        po_found = None
        gr_found = None

        # Attempt to match PO
        possible_po = po_index.get(invoice_number)
        if not possible_po:
            # fallback: match by vendor if available
            possible_po = [po for po in po_data if po.get("vendor") == vendor]
        if possible_po:
            po_found = possible_po[0]
        else:
            discrepancies.append("Missing matching PO")
            status = "flagged"

        # Attempt to match GR
        possible_gr = gr_index.get(invoice_number)
        if not possible_gr:
            # fallback: match by vendor if available
            possible_gr = [gr for gr in gr_data if gr.get("vendor") == vendor]
        if possible_gr:
            gr_found = possible_gr[0]
        else:
            discrepancies.append("Missing matching Goods Receipt")
            status = "flagged"

        # Compare amounts
        if po_found and abs(float(po_found.get("amount", amount)) - float(amount)) > 0.01:
            discrepancies.append("Amount mismatch with PO")
            status = "flagged"
        if gr_found and abs(float(gr_found.get("amount", amount)) - float(amount)) > 0.01:
            discrepancies.append("Amount mismatch with GR")
            status = "flagged"

        # (Optional) Compare line items if structure matches
        # Could add more granular line item checks here

        invoice_result = inv.copy()
        invoice_result["status"] = status
        if discrepancies:
            invoice_result["discrepancies"] = discrepancies
        validated_invoices.append(invoice_result)

    return {"validated_invoices": validated_invoices}

def enter_invoices_to_erp_tool(validated_invoices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Description:
        Enters validated invoices into the ERP system, assigns accounting codes, and logs confirmation details.

    Logic:
        1. For each validated invoice with status "validated":
            a. Assign appropriate GL codes and cost centers to each line item (simulate mapping based on line items).
            b. Create an ERP entry confirmation dict with a unique record ID, invoice number, and status.
            c. Log each confirmation.
        2. For any invoice not validated, skip entry and log as 'not entered'.
        3. Return a list of ERP entry confirmation dicts.

    Inputs:
        validated_invoices: List[Dict[str, Any]]
            List of invoices, each with 'status' (should be 'validated' to proceed).

    Outputs:
        {
            "erp_entry_confirmations": List[Dict[str, Any]]
                Each dict contains:
                    - invoice_number
                    - erp_record_id (generated)
                    - status ("entered" or "skipped")
                    - gl_codes: List[str] (assigned for entered)
                    - cost_centers: List[str] (assigned for entered)
        }

    When to use:
        After invoice validation is complete.
    """
    erp_entry_confirmations = []

    for inv in validated_invoices:
        invoice_number = inv.get("invoice_number")
        status = inv.get("status", "")
        confirmation = {
            "invoice_number": invoice_number,
        }
        if status == "validated":
            # Assign dummy GL code and cost center per line item
            line_items = inv.get("line_items", [])
            gl_codes = ["6000" for _ in line_items]  # e.g., all generic expense code
            cost_centers = ["100" for _ in line_items]
            erp_record_id = f"ERP{invoice_number}"
            confirmation.update({
                "erp_record_id": erp_record_id,
                "status": "entered",
                "gl_codes": gl_codes,
                "cost_centers": cost_centers
            })
        else:
            confirmation.update({
                "erp_record_id": None,
                "status": "skipped",
                "gl_codes": [],
                "cost_centers": []
            })
        erp_entry_confirmations.append(confirmation)

    return {"erp_entry_confirmations": erp_entry_confirmations}

def schedule_payments_for_approved_invoices_tool(approved_invoices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Description:
        Groups approved invoices into payment batches, generates payment batch file names, 
        and logs details for audit purposes.

    Logic:
        1. Sort and group approved invoices by due date (or a default if missing).
        2. Create batches where invoices with the same due date and vendor are grouped together.
        3. For each batch, generate a unique payment batch file name (deterministic).
        4. Collect the batch file names in a list.
        5. Log batch details (not returned, but could be extended later).
        6. Return the list of generated payment batch file names.

    Inputs:
        approved_invoices: List[Dict[str, Any]]
            List of approved invoice dicts, each including at least invoice_number, vendor, and due date.

    Outputs:
        {
            "payment_batch_files": List[str]
                List of unique payment batch file names (e.g., "payment_batch_20240620_ABC_Corp_1.txt").
        }

    When to use:
        After invoices are approved for payment.
    """
    from collections import defaultdict

    # Group by (due_date, vendor)
    batches = defaultdict(list)
    for inv in approved_invoices:
        due_date = inv.get("due_date", "noduedate")
        vendor = inv.get("vendor", "unknownvendor")
        key = (due_date, vendor)
        batches[key].append(inv)

    payment_batch_files = []
    for idx, ((due_date, vendor), batch_invoices) in enumerate(batches.items(), start=1):
        sanitized_vendor = vendor.replace(" ", "_")
        batch_file = f"payment_batch_{due_date}_{sanitized_vendor}_{idx}.txt"
        payment_batch_files.append(batch_file)
        # Audit log could be extended here

    return {"payment_batch_files": payment_batch_files}

def maintain_ap_records_tool(
    processed_invoices: List[Dict[str, Any]],
    payment_confirmations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Description:
        Archives processed invoice data and payment confirmations, organizes records by period and vendor,
        and generates periodic AP aging report files.

    Logic:
        1. For each processed invoice and payment confirmation:
            a. Determine the record period (e.g., YYYYMM from invoice/payment date).
            b. Determine the vendor.
            c. Assign an archive path string for each record (e.g., /archive/AP/YYYYMM/vendor/invoice_number.json).
        2. For each period, generate an AP aging report file path (e.g., /reports/AP_aging_YYYYMM.txt).
        3. Collect all archive file paths in a list.
        4. Return the list of archive file paths.

    Inputs:
        processed_invoices: List[Dict[str, Any]]
            List of processed invoice dicts.
        payment_confirmations: List[Dict[str, Any]]
            List of payment confirmation dicts.

    Outputs:
        {
            "archive_file_paths": List[str]
                List of archive file paths for invoices, payments, and AP aging reports.
        }

    When to use:
        After payment processing and at regular reporting intervals.
    """
    from collections import defaultdict

    archive_file_paths = []
    periods = set()

    # Archive processed invoices
    for inv in processed_invoices:
        date_str = inv.get("date", "nodate")
        vendor = inv.get("vendor", "unknownvendor").replace(" ", "_")
        invoice_number = inv.get("invoice_number", "noinv")
        try:
            period = date_str.replace("-", "")[:6]
        except Exception:
            period = "noperiod"
        periods.add(period)
        archive_path = f"/archive/AP/{period}/{vendor}/invoice_{invoice_number}.json"
        archive_file_paths.append(archive_path)

    # Archive payment confirmations
    for conf in payment_confirmations:
        inv_num = conf.get("invoice_number", "noinv")
        erp_id = conf.get("erp_record_id", "noerp")
        status = conf.get("status", "nostatus")
        archive_path = f"/archive/AP/payment_confirmations/payment_{inv_num}_{erp_id}_{status}.json"
        archive_file_paths.append(archive_path)

    # Generate AP aging report files for each period found
    for period in periods:
        report_path = f"/reports/AP_aging_{period}.txt"
        archive_file_paths.append(report_path)

    return {"archive_file_paths": archive_file_paths}

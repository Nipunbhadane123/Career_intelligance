import io
import csv
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)

logger = logging.getLogger(__name__)

def generate_meeting_pdf(meeting: Dict[str, Any]) -> bytes:
    """
    Task 6: High-Fidelity PDF Report Generator
    Includes:
    - Meeting details (filename, ID, platform, date, duration)
    - Executive summary
    - Key decisions
    - Action items table (with assignees, deadlines, priorities, status)
    - Participants list
    - Verified fidelity to selected meeting.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Brand Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B')
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4338CA'),
        spaceBefore=10,
        spaceAfter=6
    )
    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748B')
    )
    meta_val_style = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0F172A')
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1E293B')
    )
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0F172A')
    )

    story = []

    # 1. Header & Title Banner
    m_id = meeting.get("id", "N/A")
    m_title = meeting.get("title") or meeting.get("filename", f"Meeting #{m_id}")
    story.append(Paragraph("🧠 SynthAI — Executive Meeting Intelligence Report", ParagraphStyle('SubHeader', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#6366F1'))))
    story.append(Spacer(1, 4))
    story.append(Paragraph(m_title, title_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#6366F1'), spaceAfter=12))

    # 2. Metadata Grid
    created_at_val = str(meeting.get("created_at", "N/A"))
    if "T" in created_at_val:
        created_at_val = created_at_val.replace("T", " ")[:19]
    elif len(created_at_val) > 19:
        created_at_val = created_at_val[:19]

    duration_min = round(float(meeting.get("duration_seconds", 0)) / 60.0, 1)
    platform_name = meeting.get("platform", "upload").replace("_", " ").title()

    meta_data = [
        [
            Paragraph("Meeting ID:", meta_label_style), Paragraph(f"#{m_id}", meta_val_style),
            Paragraph("Date / Time:", meta_label_style), Paragraph(created_at_val, meta_val_style)
        ],
        [
            Paragraph("Platform:", meta_label_style), Paragraph(platform_name, meta_val_style),
            Paragraph("Est. Duration:", meta_label_style), Paragraph(f"{duration_min} mins", meta_val_style)
        ],
        [
            Paragraph("Source File:", meta_label_style), Paragraph(meeting.get("filename", "N/A"), meta_val_style),
            Paragraph("Total Participants:", meta_label_style), Paragraph(str(len(meeting.get("participants", []))), meta_val_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[80, 190, 95, 175])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#F1F5F9')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 3. Executive Summary
    story.append(Paragraph("📌 Executive Summary", h2_style))
    summary_text = meeting.get("summary") or "No executive summary available for this meeting."
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 14))

    # 4. Key Decisions
    decisions = meeting.get("key_decisions") or []
    if decisions:
        story.append(Paragraph("⚖️ Key Decisions Made", h2_style))
        for idx, dec in enumerate(decisions, 1):
            story.append(Paragraph(f"<b>{idx}.</b> {dec}", body_style))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 10))

    # 5. Action Items Table
    action_items = meeting.get("action_items") or []
    story.append(Paragraph("✅ Action Items & Deliverables", h2_style))
    if action_items:
        table_rows = [
            [
                Paragraph("#", table_header_style),
                Paragraph("Action Description", table_header_style),
                Paragraph("Assignee", table_header_style),
                Paragraph("Deadline", table_header_style),
                Paragraph("Priority", table_header_style),
                Paragraph("Status", table_header_style)
            ]
        ]
        for idx, ai in enumerate(action_items, 1):
            prio = ai.get("priority", "Medium")
            stat = ai.get("status", "Pending")
            table_rows.append([
                Paragraph(str(idx), table_cell_style),
                Paragraph(ai.get("description", ""), table_cell_style),
                Paragraph(ai.get("assigned_participant", "Unassigned"), table_cell_style),
                Paragraph(ai.get("deadline", "None") or "None", table_cell_style),
                Paragraph(prio, table_cell_style),
                Paragraph(stat, table_cell_style)
            ])

        ai_table = Table(table_rows, colWidths=[24, 220, 95, 75, 55, 71])
        ai_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4338CA')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
        ]))
        story.append(ai_table)
    else:
        story.append(Paragraph("No action items recorded for this meeting.", body_style))
    story.append(Spacer(1, 14))

    # 6. Participants List
    participants = meeting.get("participants") or []
    if participants:
        story.append(Paragraph("👥 Meeting Participants", h2_style))
        part_str = " &nbsp;•&nbsp; ".join([f"<b>{p}</b>" for p in participants])
        story.append(Paragraph(part_str, body_style))
        story.append(Spacer(1, 14))

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def generate_meeting_csv(meeting: Dict[str, Any]) -> str:
    """
    Task 6: Structured CSV Report Generator
    Produces RFC-4180 compliant CSV export containing:
    - Meeting details
    - Summary
    - Key decisions
    - Action items with assignees & deadlines
    - Participants
    """
    output = io.StringIO()
    writer = csv.writer(output)

    m_id = meeting.get("id", "N/A")
    m_title = meeting.get("title") or meeting.get("filename", "")
    created_at = str(meeting.get("created_at", ""))
    platform = meeting.get("platform", "upload")

    # Header section
    writer.writerow(["=== SynthAI Meeting Intelligence Report ==="])
    writer.writerow(["Meeting ID", m_id])
    writer.writerow(["Title", m_title])
    writer.writerow(["Filename", meeting.get("filename", "")])
    writer.writerow(["Platform", platform])
    writer.writerow(["Created At", created_at])
    writer.writerow(["Duration Seconds", meeting.get("duration_seconds", 0.0)])
    writer.writerow([])

    # Executive Summary section
    writer.writerow(["=== Executive Summary ==="])
    writer.writerow([meeting.get("summary", "")])
    writer.writerow([])

    # Participants
    writer.writerow(["=== Participants ==="])
    writer.writerow(["Participant Name"])
    for p in meeting.get("participants", []):
        writer.writerow([p])
    writer.writerow([])

    # Key Decisions
    writer.writerow(["=== Key Decisions ==="])
    writer.writerow(["#", "Decision"])
    for idx, dec in enumerate(meeting.get("key_decisions", []), 1):
        writer.writerow([idx, dec])
    writer.writerow([])

    # Action Items Table
    writer.writerow(["=== Action Items ==="])
    writer.writerow(["#", "Description", "Assigned Participant", "Deadline", "Priority", "Status"])
    for idx, ai in enumerate(meeting.get("action_items", []), 1):
        writer.writerow([
            idx,
            ai.get("description", ""),
            ai.get("assigned_participant", "Unassigned"),
            ai.get("deadline", ""),
            ai.get("priority", "Medium"),
            ai.get("status", "Pending")
        ])

    return output.getvalue()

def verify_report_contents(meeting: Dict[str, Any], report_text_or_csv: str) -> bool:
    """Task 6: Verification that generated report matches selected meeting."""
    m_id = str(meeting.get("id"))
    filename = meeting.get("filename", "")
    return (m_id in report_text_or_csv) and (filename in report_text_or_csv)

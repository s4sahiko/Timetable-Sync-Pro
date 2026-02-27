import os
import json
import base64
import time
from datetime import datetime, timedelta
import requests
import io
from PIL import Image
from pdf2image import convert_from_bytes
from flask import Flask, render_template, request, jsonify, send_file, session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24) 

# --- Constants & Configuration ---
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Current Groq vision model



# Interface for TimetableEntry (Used for type hinting and schema)
TIMETABLE_ENTRY_SCHEMA = {
    "day": {"type": "string"}, 
    "time": {"type": "string"}, 
    "subject": {"type": "string"}, 
    "location": {"type": "string"}
}

# --- Utility Functions ---

def file_to_base64(file_storage):
    """Converts a Flask FileStorage object to a Base64 string and MIME type."""
    file_bytes = file_storage.read()
    mime_type = file_storage.mimetype
    base64_string = base64.b64encode(file_bytes).decode('utf-8')
    return base64_string, mime_type

def get_next_occurrence_date(item: dict, is_start: bool) -> datetime | None:
    """Finds the next occurrence date for a given day and time."""
    days_of_week = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
    target_day_index = days_of_week.index(item['day'].lower()) if item['day'].lower() in days_of_week else -1
    
    if target_day_index == -1:
        return None

    try:
        time_str = item['time'].split('-')[0] if is_start else item['time'].split('-')[1]
        hour, minute = map(int, time_str.split(':'))
    except (IndexError, ValueError):
        return None

    now = datetime.now()
    
    # Calculate days until next target day
    today_index = now.weekday()  # Monday is 0, Sunday is 6
    # Convert to 0=Sunday, 1=Monday... to match days_of_week array
    today_index = (today_index + 1) % 7
    
    diff = (target_day_index - today_index + 7) % 7
    
    # Check if it's today and the time has already passed
    if diff == 0:
        check_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if check_time < now:
             diff = 7 # Move to next week

    date = now + timedelta(days=diff)
    date = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    return date

def format_date_to_calendar(d: datetime) -> str:
    """Formats a datetime object to YYYYMMDDTHHMMSS for iCalendar."""
    # iCalendar format: YYYYMMDDTHHMMSS 
    return d.strftime('%Y%m%dT%H%M%S')

def generate_ics_content(timetable_data: list[dict], app_id: str) -> str:
    """Generates the full iCalendar content."""
    ics_content = (
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "PRODID:-//Timetable Organizer//EN\n"
        "CALSCALE:GREGORIAN\n"
        "X-WR-CALNAME:My University Timetable\n"
    )
    
    day_map = {
        'monday': 'MO', 'tuesday': 'TU', 'wednesday': 'WE', 'thursday': 'TH', 
        'friday': 'FR', 'saturday': 'SA', 'sunday': 'SU'
    }

    dt_stamp_utc = format_date_to_calendar(datetime.utcnow()) + 'Z'

    for index, item in enumerate(timetable_data):
        day_key = item['day'].lower()
        day_to_ics = day_map.get(day_key, '')
        
        if not day_to_ics:
            continue

        startDate = get_next_occurrence_date(item, True)
        endDate = get_next_occurrence_date(item, False)
        
        if not startDate or not endDate:
            continue

        dtStart = format_date_to_calendar(startDate)
        dtEnd = format_date_to_calendar(endDate)

        # Escape commas for iCalendar fields using raw strings
        summary = item['subject'].replace(',', r'\,')
        location = item['location'].replace(',', r'\,') if item['location'] else ''

        ics_content += "BEGIN:VEVENT\n"
        ics_content += f"UID:{dtStart}-{index}@{app_id}\n"
        ics_content += f"DTSTAMP:{dt_stamp_utc}\n"
        ics_content += f"DTSTART:{dtStart}\n"
        ics_content += f"DTEND:{dtEnd}\n"
        ics_content += f"SUMMARY:{summary}\n"
        if location:
            ics_content += f"LOCATION:{location}\n"
        ics_content += f"DESCRIPTION:Generated from Timetable Organizer. Automatically repeats weekly.\n"
        ics_content += f"RRULE:FREQ=WEEKLY;BYDAY={day_to_ics}\n"
        ics_content += "END:VEVENT\n"
        
    ics_content += "END:VCALENDAR"
    return ics_content

# --- Flask Routes ---

@app.route('/')
def index():
    """Initial route to render the main template and initialize session data."""
    # Initialize session data if not present
    if 'timetable_data' not in session:
        session['timetable_data'] = []
    if 'current_step' not in session:
        session['current_step'] = 1
    
    # Pass initial data to the frontend
    return render_template('index.html', initial_data={
        'timetableData': session['timetable_data'],
        'currentStep': session['current_step']
    })

@app.route('/api/upload_and_analyze', methods=['POST'])
def upload_and_analyze():
    """Handles file upload, calls Gemini API, and stores results in session."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400
    
    if file.content_length > 4 * 1024 * 1024:
        return jsonify({'success': False, 'message': 'File size must be under 4MB.'}), 400

    if not (file.mimetype.startswith('image/') or file.mimetype == 'application/pdf'):
        return jsonify({'success': False, 'message': 'Please upload an image or PDF file.'}), 400

    try:
        image_parts = []
        if file.mimetype == 'application/pdf':
            # Convert PDF pages to images
            pdf_bytes = file.read()
            images = convert_from_bytes(pdf_bytes)
            # Limit to first 5 pages to avoid massive payloads
            for img in images[:5]:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
        else:
            base64_image, mime_type = file_to_base64(file)
            image_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            })
    except Exception as e:
        return jsonify({'success': False, 'message': f'File reading/conversion error: {str(e)}'}), 500


    # Groq API Call Logic
    api_key = GROQ_API_KEY
    if not api_key:
        return jsonify({'success': False, 'message': 'GROQ_API_KEY is not configured on the server.'}), 500

    system_instruction = (
        "You are an elite timetable parsing intelligence. Your goal is to convert any visual timetable (grid, list, or freeform) into a strict JSON format.\n\n"
        "### EXTRACTION RULES:\n"
        "1. **Day**: Identify the day of the week. Expand abbreviations (e.g., 'Mon' -> 'Monday').\n"
        "2. **Time**: Extract the time range. Convert to 24-hour format (HH:MM-HH:MM). If only a start time is given, assume a 1-hour duration. Handle formats like '9am - 10:30am' -> '09:00-10:30'.\n"
        "3. **Subject**: The core name of the class or event.\n"
        "4. **Location**: Room number, building, or digital link. Use an empty string if not found.\n"
        "5. **Empty Cells**: In grid layouts, if a cell is empty or 'No Class', ignore it.\n"
        "6. **Multi-slot**: If one entry spans multiple days or times, create separate entries for each.\n\n"
        "### OUTPUT FORMAT:\n"
        "Return ONLY a JSON object with this key: {\"timetable\": [{\"day\": \"...\", \"time\": \"...\", \"subject\": \"...\", \"location\": \"...\"}]}. "
        "Do not include any preamble, markdown formatting, or explanations."
    )
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_instruction
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the structured timetable from this image(s). Return ONLY a JSON object like: {\"timetable\": [{\"day\": \"Monday\", \"time\": \"09:00-10:30\", \"subject\": \"Math\", \"location\": \"Room 101\"}]}. Use HH:MM-HH:MM for time, full day names."
                    },
                    *image_parts
                ]
            }
        ],
        "temperature": 0
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            },
            data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json()
        
        # Grok response structure (OpenAI-compatible)
        content_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')

        if not content_text:
            raise ValueError("API response was empty or malformed.")

        # Grok might return just the array or a wrapped object depending on prompt
        # But we requested json_object, so it should be a valid JSON string.
        # Robust JSON extraction: Handle markdown blocks if the AI includes them
        import re
        json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group())
            except json.JSONDecodeError:
                # Fallback to simple load if regex extraction fails
                parsed_json = json.loads(content_text)
        else:
            parsed_json = json.loads(content_text)
        
        if isinstance(parsed_json, dict) and "timetable" in parsed_json:
            valid_data = parsed_json["timetable"]
        elif isinstance(parsed_json, list):
            valid_data = parsed_json
        else:
            valid_data = [parsed_json] if isinstance(parsed_json, dict) else []


        if not valid_data:
            return jsonify({'success': False, 'message': 'Analysis complete, but no schedule items were found.'}), 200

        # Store the raw, unsorted data in the session
        session['timetable_data'] = valid_data
        session['current_step'] = 2
        session.modified = True # Ensure session updates are saved
        
        return jsonify({
            'success': True, 
            'message': 'Analysis successful! Review and edit the extracted data.',
            'timetableData': valid_data,
            'currentStep': 2
        }), 200

    except requests.exceptions.HTTPError as e:
        error_message = f"Groq API HTTP Error: {e.response.status_code} - {e.response.text}"
        return jsonify({'success': False, 'message': error_message}), 500

    except Exception as e:
        return jsonify({'success': False, 'message': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/update_data', methods=['POST'])
def update_data():
    """Receives updated timetable data from the frontend and stores it."""
    data = request.get_json()
    new_data = data.get('timetableData', [])
    new_step = data.get('currentStep', 2)

    if not isinstance(new_data, list):
        return jsonify({'success': False, 'message': 'Invalid data format.'}), 400

    # Ensure all required keys are present (basic validation)
    for entry in new_data:
        if not all(k in entry for k in TIMETABLE_ENTRY_SCHEMA.keys()):
            return jsonify({'success': False, 'message': 'Data entry missing required fields.'}), 400

    session['timetable_data'] = new_data
    session['current_step'] = new_step
    session.modified = True
    
    return jsonify({'success': True, 'message': 'Timetable data updated.', 'currentStep': new_step}), 200


@app.route('/download_ics')
def download_ics():
    """Generates and serves the .ics file as a download attachment."""
    timetable_data = session.get('timetable_data', [])

    if not timetable_data:
        return "Timetable data is empty.", 400

    app_id = "flask-timetable-pro" 
    ics_content = generate_ics_content(timetable_data, app_id)

    temp_filename = 'timetable_schedule.ics'
    with open(temp_filename, 'w') as f:
        f.write(ics_content)
    
    response = send_file(
        temp_filename,
        mimetype='text/calendar',
        as_attachment=True,
        download_name='timetable_schedule.ics'
    )

    @response.call_on_close
    def remove_file():
        try:
            os.remove(temp_filename)
        except OSError:
            pass
            
    return response


@app.route('/open_ics')
def open_ics():
    """Serves the .ics file inline so the OS opens it with the default calendar app."""
    timetable_data = session.get('timetable_data', [])

    if not timetable_data:
        return "Timetable data is empty.", 400

    app_id = "flask-timetable-pro"
    ics_content = generate_ics_content(timetable_data, app_id)

    from flask import Response
    return Response(
        ics_content,
        mimetype='text/calendar',
        headers={
            'Content-Disposition': 'inline; filename="timetable_schedule.ics"'
        }
    )

from flask import render_template


if __name__ == '__main__':
    app.run(debug=True, port=5001)
import os
import json
import base64
import time
from datetime import datetime, timedelta
import requests
from flask import Flask, render_template, request, jsonify, send_file, session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24) 

# --- Constants & Configuration ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key="

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

    if not file.mimetype.startswith('image/'):
        return jsonify({'success': False, 'message': 'Please upload an image file.'}), 400

    try:
        base64_image, mime_type = file_to_base64(file)
    except Exception as e:
        return jsonify({'success': False, 'message': f'File reading error: {str(e)}'}), 500

    #  Gemini API Call Logic 
    api_key = GEMINI_API_KEY
    if not api_key:
        return jsonify({'success': False, 'message': 'GEMINI_API_KEY is not configured on the server.'}), 500

    system_instruction = "You are an expert timetable parser. Analyze the provided image of a time table and extract all classes/events. Return the data as a single JSON array, conforming strictly to the provided schema. The 'day' must be the full day name (e.g., 'Monday'), 'time' must be in HH:MM-HH:MM 24-hour format (e.g., '10:00-11:30'), 'subject' is the class title, and 'location' is the room/link. If any data is missing, use an empty string for that field."
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Extract the structured time table from this image. Ensure time format is HH:MM-HH:MM, day is full name, and extract only the day, time, subject, and location."},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "array", 
                "items": {
                    "type": "object", 
                    "properties": TIMETABLE_ENTRY_SCHEMA,
                    "required": list(TIMETABLE_ENTRY_SCHEMA.keys()) 
                }
            }
        }
    }

    try:
        response = requests.post(GEMINI_API_URL + api_key, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        
        json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')

        if not json_text:
            raise ValueError("API response was empty or malformed.")

        # Clean up markdown fences
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]

        parsed_data = json.loads(json_text)
        valid_data = parsed_data if isinstance(parsed_data, list) else []

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
        error_message = f"Gemini API HTTP Error: {e.response.status_code} - {e.response.text}"
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
    """Generates and serves the .ics file based on current session data."""
    timetable_data = session.get('timetable_data', [])

    if not timetable_data:
        # This case should be handled by the frontend before calling this endpoint, but we check anyway
        return "Timetable data is empty.", 400

    # Use a dummy ID for UID generation
    app_id = "flask-timetable-pro" 
    ics_content = generate_ics_content(timetable_data, app_id)

    # Save content to a temporary file
    temp_filename = 'timetable_schedule.ics'
    with open(temp_filename, 'w') as f:
        f.write(ics_content)
    
    # Send the file and clean up
    response = send_file(
        temp_filename,
        mimetype='text/calendar',
        as_attachment=True,
        download_name='timetable_schedule.ics'
    )

    # Use a response context to ensure file deletion after sending
    @response.call_on_close
    def remove_file():
        try:
            os.remove(temp_filename)
        except OSError:
            pass
            
    return response

from flask import render_template

@app.route('/about')
def about_page():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
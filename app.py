from flask import Flask, request, render_template
import os
import re
import numpy as np
import pandas as pd
from topsis.topsis import topsis  # use the class from your module
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('VERCEL') else 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def send_email_with_attachment(to_email, subject, body, attachment_path):
    EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        raise Exception('Email credentials not set in .env file')
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg.set_content(body)
    with open(attachment_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='application', subtype='octet-stream', filename=os.path.basename(attachment_path))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ''
    if request.method == 'POST':
        try:
            file = request.files['input_file']
            weights = request.form['weights']
            impacts = request.form['impacts']
            email = request.form['email']

            # Email validation
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                raise Exception("Invalid email format")

            # Save uploaded CSV
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Parse weights/impacts
            weights_list = [float(w) for w in weights.strip().split(',')]
            impacts_list = [s.strip() for s in impacts.strip().split(',')]

            # Validate counts vs CSV columns (excluding ID column)
            df_uploaded = pd.read_csv(filepath)
            if df_uploaded.shape[1] < 2:
                raise Exception("CSV must have at least 2 columns (ID + criteria).")
            expected_len = df_uploaded.shape[1] - 1
            if len(weights_list) != expected_len:
                raise Exception(f"Weights count must be {expected_len}.")
            if len(impacts_list) != expected_len:
                raise Exception(f"Impacts count must be {expected_len}.")
            if any(ch not in ['+', '-'] for ch in impacts_list):
                raise Exception("Impacts must be '+' or '-' only.")

            # Run TOPSIS
            t = topsis(filepath, weights_list, impacts_list)
            t.topsis_main(debug=False)  # computes p_scores and prints summary

            # Build result CSV (ID, Topsis Score, Rank)
            first_col_name = df_uploaded.columns[0]
            final_scores_sorted = np.argsort(t.p_scores)
            max_index = len(final_scores_sorted)
            ranks = [max_index - np.where(final_scores_sorted == i)[0][0] for i in range(max_index)]

            output_df = pd.DataFrame({
                first_col_name: t.df_copy_id,
                "Topsis Score": t.p_scores,
                "Rank": ranks
            })

            # Save result and email
            base = os.path.splitext(file.filename)[0]
            output_path = os.path.join(UPLOAD_FOLDER, f"result_{base}.csv")
            output_df.to_csv(output_path, index=False)

            send_email_with_attachment(email, 'TOPSIS Result', 'Find attached your TOPSIS result.', output_path)
            message = 'Result sent to your email!'
        except Exception as e:
            message = f'Error: {e}'
    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)



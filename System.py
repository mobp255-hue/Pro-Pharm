# ============================================================================
# PHARMACY MANAGEMENT SYSTEM – FINAL PRODUCTION EDITION
# ============================================================================
# - Auto‑handles missing optional libraries (Prophet, face_recognition)
# - Robust database connection with pooling and retry
# - All features: Inventory, POS, Customers, Prescriptions, Suppliers,
#   Purchase Orders, Stock Adjustments, Sales Returns, Reports, Settings
# - Premium responsive UI, digital stamp, biometric support (if installed)
# - Works with st.secrets (Streamlit Cloud) or .env (local)
# ============================================================================

import streamlit as st
import psycopg2
from psycopg2 import pool, sql, extras
import pandas as pd
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
import io
import base64
from io import BytesIO
import hmac
import time
import re

# PDF & barcode – these are always required
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import barcode
from barcode.writer import ImageWriter

# --- Optional libraries: gracefully handle missing ---
try:
    import face_recognition
    import cv2
    BIOMETRIC_AVAILABLE = True
except ImportError:
    BIOMETRIC_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# sklearn is usually installed (required for fallback forecasting)
from sklearn.linear_model import LinearRegression
import numpy as np

# ============================================================================
# 1. DATABASE CONNECTION (with st.secrets fallback)
# ============================================================================

def get_db_params():
    """Return database connection parameters from st.secrets (cloud) or .env (local)."""
    try:
        # On Streamlit Cloud, secrets are available
        secrets = st.secrets
        if "supabase" in secrets:
            db = secrets["supabase"]
            return {
                "host": db["host"],
                "port": db["port"],
                "dbname": db["database"],
                "user": db["user"],
                "password": db["password"],
                "sslmode": db.get("ssl", "require")
            }
    except (AttributeError, KeyError):
        pass

    # Fallback to .env for local development
    from dotenv import load_dotenv
    load_dotenv()
    return {
        "host": os.getenv("SUPABASE_HOST", "localhost"),
        "port": os.getenv("SUPABASE_PORT", "5432"),
        "dbname": os.getenv("SUPABASE_DATABASE", "postgres"),
        "user": os.getenv("SUPABASE_USER", "postgres"),
        "password": os.getenv("SUPABASE_PASSWORD", ""),
        "sslmode": os.getenv("SUPABASE_SSL", "require")
    }

# Get parameters and create connection pool
params = get_db_params()
if not params["password"]:
    st.error("❌ Database password not set. Please set SUPABASE_PASSWORD in .env or st.secrets.")
    st.stop()

# Retry logic for connection pool
max_retries = 3
retry_delay = 2
for attempt in range(max_retries):
    try:
        connection_pool = pool.SimpleConnectionPool(
            1, 20,
            host=params["host"],
            port=params["port"],
            dbname=params["dbname"],
            user=params["user"],
            password=params["password"],
            sslmode=params["sslmode"]
        )
        break
    except Exception as e:
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
        else:
            st.error(f"❌ Database connection failed after {max_retries} attempts: {e}")
            st.stop()

def get_db_connection():
    """Get a connection from the pool."""
    return connection_pool.getconn()

def return_db_connection(conn):
    """Return connection to the pool."""
    connection_pool.putconn(conn)

# ============================================================================
# 2. SECURITY & HELPERS
# ============================================================================

def get_secret_key():
    try:
        return st.secrets["security"]["secret_key"]
    except (AttributeError, KeyError):
        return os.getenv("SECRET_KEY", "default-secret-change-me")

SECRET_KEY = get_secret_key()

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _verify_biometric(face_encoding_json: str, image) -> bool:
    if not BIOMETRIC_AVAILABLE or not face_encoding_json:
        return False
    try:
        stored_encoding = json.loads(face_encoding_json)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return False
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if not encodings:
            return False
        return face_recognition.compare_faces([stored_encoding], encodings[0])[0]
    except Exception:
        return False

def _get_company_data() -> dict:
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    cur.execute("SELECT * FROM company_settings LIMIT 1")
    row = cur.fetchone()
    cur.close()
    return_db_connection(conn)
    if not row:
        return {}
    data = dict(row)
    stamp_data = f"{data['company_name']}|{data['address']}|{data['phone']}|{data['tax_id']}"
    expected = hmac.new(SECRET_KEY.encode(), stamp_data.encode(), hashlib.sha256).hexdigest()
    data['stamp_valid'] = (data.get('digital_stamp') == expected)
    return data

def _update_company_data(data: dict) -> bool:
    conn = get_db_connection()
    cur = conn.cursor()
    stamp_data = f"{data['company_name']}|{data['address']}|{data['phone']}|{data['tax_id']}"
    stamp = hmac.new(SECRET_KEY.encode(), stamp_data.encode(), hashlib.sha256).hexdigest()
    cur.execute("""
        UPDATE company_settings
        SET company_name=%s, address=%s, phone=%s, email=%s, tax_id=%s,
            currency_symbol=%s, receipt_footer=%s, logo_path=%s, digital_stamp=%s
        WHERE setting_id = 1
    """, (data['company_name'], data['address'], data['phone'],
          data['email'], data['tax_id'], data['currency_symbol'],
          data['receipt_footer'], data['logo_path'], stamp))
    conn.commit()
    cur.close()
    return_db_connection(conn)
    return True

def get_logo_base64() -> str:
    company = _get_company_data()
    logo_path = company.get('logo_path', '')
    if logo_path and os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{encoded}"
    return ""

def log_activity(user_id, action, details=""):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO activity_logs (user_id, action, details) VALUES (%s, %s, %s)",
                (user_id, action, details))
    conn.commit()
    cur.close()
    return_db_connection(conn)

def require_role(roles):
    if not st.session_state.logged_in:
        st.error("Please login.")
        st.stop()
    if st.session_state.user['role'] not in roles:
        st.error(f"Access denied. Required: {', '.join(roles)}")
        st.stop()

# ============================================================================
# 3. DATABASE SCHEMA & MIGRATION (PostgreSQL)
# ============================================================================

def column_exists(conn, table, column):
    cur = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s
        )
    """, (table, column))
    exists = cur.fetchone()[0]
    cur.close()
    return exists

def init_db():
    """Create tables and perform schema migrations."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Create tables (IF NOT EXISTS)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT CHECK(role IN ('admin','pharmacist','salesman','manager')) DEFAULT 'salesman',
            full_name TEXT,
            email TEXT,
            phone TEXT,
            face_encoding TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS company_settings (
            setting_id SERIAL PRIMARY KEY,
            company_name TEXT,
            address TEXT,
            phone TEXT,
            email TEXT,
            tax_id TEXT,
            currency_symbol TEXT DEFAULT '$',
            receipt_footer TEXT,
            logo_path TEXT,
            digital_stamp TEXT
        )
    """)
    cur.execute("SELECT * FROM company_settings")
    if not cur.fetchone():
        default_data = {
            'company_name': 'My Pharmacy',
            'address': '123 Main St, City',
            'phone': '+1-234-567-890',
            'email': 'info@pharmacy.com',
            'tax_id': 'TAX-12345',
            'currency_symbol': '$',
            'receipt_footer': 'Thank you for your business!',
            'logo_path': '',
        }
        stamp_data = f"{default_data['company_name']}|{default_data['address']}|{default_data['phone']}|{default_data['tax_id']}"
        stamp = hmac.new(SECRET_KEY.encode(), stamp_data.encode(), hashlib.sha256).hexdigest()
        cur.execute("""
            INSERT INTO company_settings (
                company_name, address, phone, email, tax_id,
                currency_symbol, receipt_footer, logo_path, digital_stamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (default_data['company_name'], default_data['address'],
              default_data['phone'], default_data['email'], default_data['tax_id'],
              default_data['currency_symbol'], default_data['receipt_footer'],
              default_data['logo_path'], stamp))

    # Medicines
    cur.execute("""
        CREATE TABLE IF NOT EXISTS medicines (
            medicine_id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            generic_name TEXT,
            category TEXT,
            manufacturer TEXT,
            supplier_id INTEGER REFERENCES suppliers(supplier_id),
            batch_number TEXT,
            lot_number TEXT,
            quantity INTEGER DEFAULT 0,
            min_stock_level INTEGER DEFAULT 10,
            unit_price REAL,
            selling_price REAL,
            expiry_date DATE,
            is_prescription_required BOOLEAN DEFAULT FALSE,
            barcode TEXT,
            location TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medicines_name ON medicines(name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medicines_barcode ON medicines(barcode)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medicines_category ON medicines(category)")

    # Suppliers
    cur.execute("""
        CREATE TABLE IF NOT EXISTS suppliers (
            supplier_id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            contact_person TEXT,
            phone TEXT,
            email TEXT,
            address TEXT,
            tax_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Customers
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id SERIAL PRIMARY KEY,
            full_name TEXT NOT NULL,
            date_of_birth DATE,
            gender TEXT,
            phone TEXT,
            email TEXT,
            address TEXT,
            medical_conditions TEXT,
            allergies TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_customers_name ON customers(full_name)")

    # Sales
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            sale_id SERIAL PRIMARY KEY,
            invoice_number TEXT UNIQUE NOT NULL,
            customer_id INTEGER REFERENCES customers(customer_id),
            user_id INTEGER REFERENCES users(user_id),
            total_amount REAL,
            discount REAL DEFAULT 0,
            tax REAL DEFAULT 0,
            payment_method TEXT CHECK(payment_method IN ('cash','card','insurance','other')),
            payment_status TEXT DEFAULT 'completed',
            sale_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date)")

    # Sale Items
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sale_items (
            sale_item_id SERIAL PRIMARY KEY,
            sale_id INTEGER REFERENCES sales(sale_id),
            medicine_id INTEGER REFERENCES medicines(medicine_id),
            quantity INTEGER,
            unit_price REAL,
            total_price REAL
        )
    """)

    # Sales Returns
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales_returns (
            return_id SERIAL PRIMARY KEY,
            sale_id INTEGER REFERENCES sales(sale_id),
            return_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reason TEXT,
            total_refund REAL,
            user_id INTEGER REFERENCES users(user_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS return_items (
            return_item_id SERIAL PRIMARY KEY,
            return_id INTEGER REFERENCES sales_returns(return_id),
            medicine_id INTEGER REFERENCES medicines(medicine_id),
            quantity INTEGER,
            unit_price REAL
        )
    """)

    # Prescriptions
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            prescription_id SERIAL PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(customer_id),
            doctor_name TEXT,
            issued_date DATE,
            expiry_date DATE,
            notes TEXT,
            status TEXT DEFAULT 'active'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prescription_items (
            prescription_item_id SERIAL PRIMARY KEY,
            prescription_id INTEGER REFERENCES prescriptions(prescription_id),
            medicine_id INTEGER REFERENCES medicines(medicine_id),
            dosage TEXT,
            frequency TEXT,
            duration TEXT
        )
    """)

    # Purchase Orders
    cur.execute("""
        CREATE TABLE IF NOT EXISTS purchase_orders (
            order_id SERIAL PRIMARY KEY,
            supplier_id INTEGER REFERENCES suppliers(supplier_id),
            order_date DATE,
            expected_delivery DATE,
            status TEXT DEFAULT 'pending',
            total_amount REAL,
            notes TEXT,
            created_by INTEGER REFERENCES users(user_id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS purchase_order_items (
            order_item_id SERIAL PRIMARY KEY,
            order_id INTEGER REFERENCES purchase_orders(order_id),
            medicine_id INTEGER REFERENCES medicines(medicine_id),
            quantity INTEGER,
            unit_price REAL,
            total_price REAL,
            received_quantity INTEGER DEFAULT 0
        )
    """)

    # Stock Movements
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_movements (
            movement_id SERIAL PRIMARY KEY,
            medicine_id INTEGER REFERENCES medicines(medicine_id),
            quantity_change INTEGER,
            movement_type TEXT CHECK(movement_type IN ('purchase','sale','return','adjustment','adjustment_in','adjustment_out')),
            reference_id INTEGER,
            user_id INTEGER REFERENCES users(user_id),
            notes TEXT,
            movement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stock_movements_medicine ON stock_movements(medicine_id)")

    # Activity Logs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            log_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id),
            action TEXT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_activity_logs_timestamp ON activity_logs(timestamp)")

    # Insert admin if not exists
    admin_hash = _hash_password("admin123")
    cur.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cur.fetchone():
        cur.execute("""
            INSERT INTO users (username, password_hash, role, full_name, email, is_active)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, ('admin', admin_hash, 'admin', 'Administrator', 'admin@pharmacy.com', True))

    # --- Migration: check if any columns are missing and add them ---
    if not column_exists(conn, 'medicines', 'location'):
        cur.execute("ALTER TABLE medicines ADD COLUMN location TEXT")
    if not column_exists(conn, 'users', 'face_encoding'):
        cur.execute("ALTER TABLE users ADD COLUMN face_encoding TEXT")
    if not column_exists(conn, 'company_settings', 'logo_path'):
        cur.execute("ALTER TABLE company_settings ADD COLUMN logo_path TEXT")
    if not column_exists(conn, 'company_settings', 'digital_stamp'):
        cur.execute("ALTER TABLE company_settings ADD COLUMN digital_stamp TEXT")

    conn.commit()
    cur.close()
    return_db_connection(conn)

# ============================================================================
# 4. CACHED QUERIES (PostgreSQL)
# ============================================================================

@st.cache_data(ttl=60)
def get_sales_summary(days=30):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT date(sale_date) as date, COUNT(*) as sales_count, SUM(total_amount) as revenue
        FROM sales
        WHERE sale_date >= CURRENT_DATE - INTERVAL '1 day' * %s
        GROUP BY date(sale_date)
        ORDER BY date
    """, (days,))
    rows = cur.fetchall()
    cur.close()
    return_db_connection(conn)
    if rows:
        return pd.DataFrame(rows, columns=['date', 'sales_count', 'revenue'])
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_inventory_summary():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT category, COUNT(*) as count, SUM(quantity) as total_stock
        FROM medicines
        WHERE is_active = TRUE
        GROUP BY category
    """)
    rows = cur.fetchall()
    cur.close()
    return_db_connection(conn)
    if rows:
        return pd.DataFrame(rows, columns=['category', 'count', 'total_stock'])
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_low_stock_items():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT name, quantity, min_stock_level
        FROM medicines
        WHERE quantity <= min_stock_level AND quantity > 0 AND is_active = TRUE
        ORDER BY quantity
    """)
    rows = cur.fetchall()
    cur.close()
    return_db_connection(conn)
    if rows:
        return pd.DataFrame(rows, columns=['name', 'quantity', 'min_stock_level'])
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_expiring_items():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT name, expiry_date, quantity
        FROM medicines
        WHERE expiry_date <= CURRENT_DATE + INTERVAL '30 days'
          AND expiry_date >= CURRENT_DATE
          AND is_active = TRUE
        ORDER BY expiry_date
    """)
    rows = cur.fetchall()
    cur.close()
    return_db_connection(conn)
    if rows:
        return pd.DataFrame(rows, columns=['name', 'expiry_date', 'quantity'])
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_suppliers_dict():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT supplier_id, name FROM suppliers")
    rows = cur.fetchall()
    cur.close()
    return_db_connection(conn)
    return {row[0]: row[1] for row in rows}

@st.cache_data(ttl=300)
def get_categories():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT category FROM medicines WHERE is_active = TRUE")
    rows = cur.fetchall()
    cur.close()
    return_db_connection(conn)
    base = ['Analgesic','Antibiotic','Antihistamine','Antidepressant','Antiviral','Cardiovascular','Dermatological','Gastrointestinal','Nutritional','Respiratory','Other']
    all_cats = base + [row[0] for row in rows if row[0]]
    return sorted(set(all_cats))

# ============================================================================
# 5. STREAMLIT APP CONFIG
# ============================================================================

st.set_page_config(
    page_title="💊 Pharmacy Pro",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logo_base64 = get_logo_base64()
if logo_base64:
    st.sidebar.image(logo_base64, width=200)

# Premium UI custom CSS
st.markdown("""
    <style>
        .reportview-container .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        .stButton button { width: 100%; border-radius: 8px; font-weight: 500; }
        .stMetric { background: linear-gradient(145deg, #ffffff, #f0f2f6); border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .stMetric .stMetricValue { font-size: 28px !important; font-weight: 700; }
        .stMetric .stMetricDelta { font-weight: 500; }
        .stDataFrame { border-radius: 8px; overflow: hidden; }
        .stSidebar .stImage { margin-top: 10px; }
        @media (max-width: 600px) { .stColumns { flex-direction: column !important; } }
        .stAlert { margin-top: 0.5rem; border-radius: 8px; }
        h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }
    </style>
""", unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'use_biometric' not in st.session_state:
    st.session_state.use_biometric = BIOMETRIC_AVAILABLE

# ============================================================================
# 6. LOGIN PAGE
# ============================================================================

def login_page():
    st.title("💊 Pharmacy Pro")
    st.subheader("Secure Login")

    if BIOMETRIC_AVAILABLE and st.session_state.use_biometric:
        st.info("🔐 Biometric login enabled (face recognition).")
        if st.button("📸 Scan Face"):
            uploaded_file = st.file_uploader("Upload a face image for verification", type=["jpg", "png"])
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                conn = get_db_connection()
                cur = conn.cursor(cursor_factory=extras.RealDictCursor)
                cur.execute("SELECT user_id, username, role, full_name, face_encoding FROM users WHERE is_active = TRUE")
                users = cur.fetchall()
                cur.close()
                return_db_connection(conn)
                authenticated = False
                for user in users:
                    if user['face_encoding']:
                        if _verify_biometric(user['face_encoding'], img):
                            st.session_state.logged_in = True
                            st.session_state.user = {
                                "user_id": user['user_id'],
                                "role": user['role'],
                                "full_name": user['full_name']
                            }
                            authenticated = True
                            log_activity(user['user_id'], "Login", "Biometric")
                            st.success(f"Welcome {user['full_name']}!")
                            st.experimental_rerun()
                            break
                if not authenticated:
                    st.error("Face not recognized.")
    else:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=extras.RealDictCursor)
            cur.execute("SELECT user_id, password_hash, role, full_name, is_active FROM users WHERE username = %s", (username,))
            user = cur.fetchone()
            cur.close()
            return_db_connection(conn)
            if user and user['password_hash'] == _hash_password(password) and user['is_active']:
                st.session_state.logged_in = True
                st.session_state.user = {
                    "user_id": user['user_id'],
                    "role": user['role'],
                    "full_name": user['full_name']
                }
                log_activity(user['user_id'], "Login", "Password")
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials or inactive account")

# ============================================================================
# 7. PDF GENERATORS
# ============================================================================

def generate_simple_pdf(df, title="Report"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>{title}</b>", styles['Title']))
    story.append(Spacer(1, 12))
    if not df.empty:
        table_data = [df.columns.tolist()] + df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

def generate_receipt(invoice, sale_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    cur.execute("""
        SELECT s.*, u.full_name as cashier, c.full_name as customer
        FROM sales s
        LEFT JOIN users u ON s.user_id = u.user_id
        LEFT JOIN customers c ON s.customer_id = c.customer_id
        WHERE s.sale_id = %s
    """, (sale_id,))
    sale = cur.fetchone()
    if not sale:
        cur.close()
        return_db_connection(conn)
        return None
    cur.execute("""
        SELECT si.*, m.name
        FROM sale_items si
        JOIN medicines m ON si.medicine_id = m.medicine_id
        WHERE si.sale_id = %s
    """, (sale_id,))
    items = cur.fetchall()
    cur.close()
    return_db_connection(conn)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    company = _get_company_data()
    company_name = company.get('company_name', 'Pharmacy')
    address = company.get('address', '')
    phone = company.get('phone', '')
    currency = company.get('currency_symbol', '$')

    story.append(Paragraph(f"<b>{company_name}</b>", styles['Title']))
    story.append(Paragraph(address, styles['Normal']))
    story.append(Paragraph(f"Phone: {phone}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>INVOICE: {invoice}</b>", styles['Heading2']))
    story.append(Paragraph(f"Date: {sale['sale_date']}", styles['Normal']))
    story.append(Paragraph(f"Cashier: {sale['cashier'] if sale['cashier'] else 'N/A'}", styles['Normal']))
    story.append(Paragraph(f"Customer: {sale['customer'] if sale['customer'] else 'Walk-in'}", styles['Normal']))
    story.append(Spacer(1, 12))

    table_data = [["Item", "Qty", "Unit Price", "Total"]]
    for item in items:
        table_data.append([item['name'], str(item['quantity']), f"{currency}{item['unit_price']:.2f}", f"{currency}{item['quantity']*item['unit_price']:.2f}"])
    table = Table(table_data, colWidths=[3*inch, 0.75*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Subtotal: {currency}{sale['total_amount']:.2f}</b>", styles['Normal']))
    if sale['discount'] > 0:
        story.append(Paragraph(f"Discount: -{currency}{sale['discount']:.2f}", styles['Normal']))
    if sale['tax'] > 0:
        story.append(Paragraph(f"Tax: +{currency}{sale['tax']:.2f}", styles['Normal']))
    story.append(Paragraph(f"<b>Total: {currency}{sale['total_amount']:.2f}</b>", styles['Heading2']))
    story.append(Spacer(1, 12))
    if company.get('receipt_footer'):
        story.append(Paragraph(company['receipt_footer'], styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# ============================================================================
# 8. DASHBOARD
# ============================================================================

def show_dashboard():
    st.header("📊 Dashboard")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

    df_sales = get_sales_summary(30)
    total_revenue = df_sales['revenue'].sum() if not df_sales.empty else 0
    total_sales = df_sales['sales_count'].sum() if not df_sales.empty else 0
    low_stock = len(get_low_stock_items())
    expiring = len(get_expiring_items())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Revenue (30d)", f"${total_revenue:,.2f}", delta=f"{total_sales} sales")
    col2.metric("🧾 Sales (30d)", total_sales)
    col3.metric("⚠️ Low Stock", low_stock, delta="Reorder" if low_stock > 0 else "OK")
    col4.metric("⏳ Expiring Soon", expiring, delta="Check" if expiring > 0 else "OK")

    if not df_sales.empty:
        fig = px.line(df_sales, x='date', y='revenue', title='Daily Revenue (30 days)')
        st.plotly_chart(fig, use_container_width=True)

    df_inv = get_inventory_summary()
    if not df_inv.empty:
        fig2 = px.pie(df_inv, values='count', names='category', title='Inventory by Category')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("⚠️ Low Stock Alerts")
    low_df = get_low_stock_items()
    if not low_df.empty:
        st.dataframe(low_df, use_container_width=True)
    else:
        st.success("All items are well stocked.")

    st.subheader("⏳ Expiring Soon (30 days)")
    exp_df = get_expiring_items()
    if not exp_df.empty:
        st.dataframe(exp_df, use_container_width=True)
    else:
        st.success("No items expiring in the next 30 days.")

    st.subheader("⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    if col1.button("➕ New Sale"):
        st.session_state.menu = "Sales (POS)"
        st.experimental_rerun()
    if col2.button("📦 Add Medicine"):
        st.session_state.menu = "Inventory"
        st.experimental_rerun()
    if col3.button("👤 New Customer"):
        st.session_state.menu = "Customers"
        st.experimental_rerun()

# ============================================================================
# 9. INVENTORY (full CRUD)
# ============================================================================

def show_inventory():
    st.header("📦 Inventory Management")
    require_role(["admin", "pharmacist", "manager"])
    conn = get_db_connection()

    tab1, tab2, tab3, tab4 = st.tabs(["📋 View", "➕ Add", "🏷️ Labels", "📊 Export"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            search = st.text_input("🔍 Search (name, generic, barcode)")
        with col2:
            cat_filter = st.selectbox("Category", ["All"] + get_categories())
        with col3:
            stock_filter = st.selectbox("Stock Status", ["All", "In Stock", "Low Stock", "Out of Stock"])

        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        query = "SELECT * FROM medicines WHERE is_active = TRUE"
        params = []
        if search:
            query += " AND (name ILIKE %s OR generic_name ILIKE %s OR barcode = %s)"
            s = f"%{search}%"
            params.extend([s, s, search])
        if cat_filter != "All":
            query += " AND category = %s"
            params.append(cat_filter)
        if stock_filter == "Low Stock":
            query += " AND quantity <= min_stock_level AND quantity > 0"
        elif stock_filter == "Out of Stock":
            query += " AND quantity = 0"
        elif stock_filter == "In Stock":
            query += " AND quantity > min_stock_level"
        query += " ORDER BY name"
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        if not rows:
            st.info("No medicines found.")
        else:
            df = pd.DataFrame(rows)
            st.dataframe(df[['medicine_id','name','category','quantity','selling_price','expiry_date','min_stock_level']],
                         use_container_width=True)

            with st.expander("✏️ Edit or Delete Medicine"):
                med_id = st.number_input("Medicine ID", min_value=1, step=1)
                if med_id:
                    cur = conn.cursor(cursor_factory=extras.RealDictCursor)
                    cur.execute("SELECT * FROM medicines WHERE medicine_id = %s", (med_id,))
                    med = cur.fetchone()
                    cur.close()
                    if med:
                        col1, col2 = st.columns(2)
                        with col1:
                            new_name = st.text_input("Name", med['name'])
                            new_qty = st.number_input("Quantity", value=int(med['quantity']), step=1)
                            new_price = st.number_input("Selling Price", value=float(med['selling_price']), step=0.5)
                            new_min = st.number_input("Min Stock Level", value=int(med['min_stock_level']), step=1)
                        with col2:
                            new_expiry = st.date_input("Expiry Date", value=pd.to_datetime(med['expiry_date']).date())
                            new_barcode = st.text_input("Barcode", med['barcode'] or "")
                            new_location = st.text_input("Location", med['location'] or "")
                        if st.button("Update Medicine"):
                            cur2 = conn.cursor()
                            cur2.execute("""
                                UPDATE medicines
                                SET name=%s, quantity=%s, selling_price=%s, min_stock_level=%s,
                                    expiry_date=%s, barcode=%s, location=%s
                                WHERE medicine_id=%s
                            """, (new_name, new_qty, new_price, new_min, new_expiry,
                                  new_barcode, new_location, med_id))
                            conn.commit()
                            cur2.close()
                            st.success("Updated!")
                            log_activity(st.session_state.user["user_id"], "Update Medicine", f"ID {med_id}")
                            st.cache_data.clear()
                            st.experimental_rerun()
                        if st.button("Delete (Deactivate)", type="primary"):
                            cur2 = conn.cursor()
                            cur2.execute("UPDATE medicines SET is_active = FALSE WHERE medicine_id = %s", (med_id,))
                            conn.commit()
                            cur2.close()
                            st.warning("Deactivated!")
                            st.cache_data.clear()
                            st.experimental_rerun()
                    else:
                        st.error("Not found")

    with tab2:
        with st.form("add_medicine_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name*")
                generic = st.text_input("Generic Name")
                category = st.selectbox("Category", get_categories())
                manufacturer = st.text_input("Manufacturer")
                supplier_id = st.selectbox("Supplier", list(get_suppliers_dict().keys()), format_func=lambda x: get_suppliers_dict().get(x, x))
                batch = st.text_input("Batch Number")
                lot = st.text_input("Lot Number")
            with col2:
                qty = st.number_input("Quantity", min_value=0, step=1)
                min_stock = st.number_input("Min Stock Level", value=10, step=1)
                unit_price = st.number_input("Unit Price ($)", min_value=0.0, step=0.5)
                selling_price = st.number_input("Selling Price ($)*", min_value=0.0, step=0.5)
                expiry = st.date_input("Expiry Date")
                prescription = st.checkbox("Prescription Required")
                barcode = st.text_input("Barcode (optional)")
                location = st.text_input("Shelf/Rack Location")
            submitted = st.form_submit_button("Add Medicine")
            if submitted and name and selling_price:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO medicines (
                        name, generic_name, category, manufacturer, supplier_id,
                        batch_number, lot_number, quantity, min_stock_level,
                        unit_price, selling_price, expiry_date,
                        is_prescription_required, barcode, location
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (name, generic, category, manufacturer, supplier_id,
                      batch, lot, qty, min_stock, unit_price, selling_price,
                      expiry, prescription, barcode, location))
                conn.commit()
                cur.close()
                st.success(f"Added {name}")
                log_activity(st.session_state.user["user_id"], "Add Medicine", name)
                st.cache_data.clear()
                st.experimental_rerun()

    with tab3:
        st.subheader("🏷️ Generate Barcode Labels")
        med_id_label = st.number_input("Medicine ID for label", min_value=1, step=1)
        if med_id_label:
            cur = conn.cursor(cursor_factory=extras.RealDictCursor)
            cur.execute("SELECT name, barcode FROM medicines WHERE medicine_id = %s", (med_id_label,))
            med = cur.fetchone()
            cur.close()
            if med and med['barcode']:
                try:
                    code = barcode.get_barcode_class('code128')
                    code_obj = code(med['barcode'], writer=ImageWriter())
                    buffer = BytesIO()
                    code_obj.write(buffer)
                    buffer.seek(0)
                    st.image(buffer, caption=f"Barcode for {med['name']}", use_column_width=False)
                    st.download_button("Download Barcode", data=buffer, file_name=f"{med['name']}_barcode.png", mime="image/png")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("No barcode set for this medicine.")

    with tab4:
        st.subheader("📊 Export Inventory")
        if st.button("Export to CSV"):
            cur = conn.cursor(cursor_factory=extras.RealDictCursor)
            cur.execute("SELECT * FROM medicines WHERE is_active = TRUE ORDER BY name")
            rows = cur.fetchall()
            cur.close()
            df_export = pd.DataFrame(rows)
            csv = df_export.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="inventory.csv", mime="text/csv")
        if st.button("Export to PDF (simple)"):
            cur = conn.cursor(cursor_factory=extras.RealDictCursor)
            cur.execute("SELECT name, quantity, selling_price, expiry_date FROM medicines WHERE is_active = TRUE ORDER BY name")
            rows = cur.fetchall()
            cur.close()
            if rows:
                df_pdf = pd.DataFrame(rows)
                pdf = generate_simple_pdf(df_pdf, "Inventory Report")
                st.download_button("Download PDF", data=pdf, file_name="inventory_report.pdf", mime="application/pdf")

    return_db_connection(conn)

# ============================================================================
# 10. POINT OF SALE (POS)
# ============================================================================

def show_pos():
    st.header("🛒 Point of Sale")
    require_role(["admin", "pharmacist", "salesman"])
    conn = get_db_connection()

    # Customer selection
    cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    cur.execute("SELECT customer_id, full_name FROM customers WHERE is_active = TRUE ORDER BY full_name")
    customers = cur.fetchall()
    cur.close()
    customer_choices = {0: "Walk-in Customer"}
    for c in customers:
        customer_choices[c['customer_id']] = c['full_name']
    selected_customer = st.selectbox("Customer", list(customer_choices.keys()), format_func=lambda x: customer_choices.get(x, "Unknown"))

    # Search medicine
    search = st.text_input("🔍 Search Medicine (name or barcode)")
    if search:
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cur.execute("""
            SELECT medicine_id, name, selling_price, quantity
            FROM medicines
            WHERE (name ILIKE %s OR barcode = %s) AND quantity > 0 AND is_active = TRUE
        """, (f"%{search}%", search))
        results = cur.fetchall()
        cur.close()
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            med_id = st.number_input("Enter Medicine ID", min_value=1, step=1)
            qty = st.number_input("Quantity", min_value=1, step=1)
            if st.button("Add to Cart"):
                cur = conn.cursor(cursor_factory=extras.RealDictCursor)
                cur.execute("SELECT quantity, selling_price FROM medicines WHERE medicine_id = %s", (med_id,))
                med = cur.fetchone()
                cur.close()
                if med and med['quantity'] >= qty:
                    name = next((r['name'] for r in results if r['medicine_id'] == med_id), None)
                    if name:
                        st.session_state.cart.append({
                            "medicine_id": med_id,
                            "name": name,
                            "quantity": qty,
                            "unit_price": med['selling_price'],
                            "total": med['selling_price'] * qty
                        })
                        st.success("Added to cart!")
                    else:
                        st.error("Medicine not found in search results.")
                else:
                    st.error("Insufficient stock or not found.")

    st.subheader("🛍️ Current Cart")
    if st.session_state.cart:
        cart_df = pd.DataFrame(st.session_state.cart)
        st.dataframe(cart_df[['name','quantity','unit_price','total']], use_container_width=True)
        total = sum(item['total'] for item in st.session_state.cart)

        discount_pct = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, step=1.0)
        tax_pct = st.number_input("Tax (%)", min_value=0.0, max_value=30.0, step=1.0, value=0.0)
        discount_amount = total * (discount_pct / 100)
        tax_amount = (total - discount_amount) * (tax_pct / 100)
        final_total = total - discount_amount + tax_amount
        st.metric("Total After Discount & Tax", f"${final_total:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Cart"):
                st.session_state.cart = []
                st.experimental_rerun()
        with col2:
            payment_method = st.selectbox("Payment Method", ["cash", "card", "insurance", "other"])
            if st.button("Checkout"):
                process_checkout(selected_customer, final_total, discount_amount, tax_amount, payment_method)
    else:
        st.info("Cart is empty.")
    return_db_connection(conn)

def process_checkout(customer_id, total, discount, tax, payment_method):
    conn = get_db_connection()
    cur = conn.cursor()
    invoice = f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    user_id = st.session_state.user["user_id"]

    cur.execute("""
        INSERT INTO sales (invoice_number, customer_id, user_id, total_amount, discount, tax, payment_method)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (invoice, customer_id, user_id, total, discount, tax, payment_method))
    sale_id = cur.lastrowid

    for item in st.session_state.cart:
        cur.execute("""
            INSERT INTO sale_items (sale_id, medicine_id, quantity, unit_price, total_price)
            VALUES (%s, %s, %s, %s, %s)
        """, (sale_id, item['medicine_id'], item['quantity'], item['unit_price'], item['total']))
        cur.execute("UPDATE medicines SET quantity = quantity - %s WHERE medicine_id = %s",
                    (item['quantity'], item['medicine_id']))
        cur.execute("""
            INSERT INTO stock_movements (medicine_id, quantity_change, movement_type, reference_id, user_id, notes)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (item['medicine_id'], -item['quantity'], 'sale', sale_id, user_id, f"Invoice {invoice}"))

    conn.commit()
    cur.close()
    return_db_connection(conn)

    receipt_pdf = generate_receipt(invoice, sale_id)
    st.success(f"✅ Sale completed! Invoice: {invoice}")
    log_activity(user_id, "Sale", f"Invoice {invoice} total ${total:.2f}")
    st.download_button("📄 Download Receipt (PDF)", data=receipt_pdf, file_name=f"{invoice}.pdf", mime="application/pdf")
    st.balloons()
    st.session_state.cart = []
    st.cache_data.clear()
    st.experimental_rerun()

# ============================================================================
# 11. CUSTOMERS
# ============================================================================

def show_customers():
    st.header("👤 Customer Management")
    require_role(["admin", "pharmacist", "manager", "salesman"])
    conn = get_db_connection()

    tab1, tab2 = st.tabs(["📋 List", "➕ Add"])

    with tab1:
        search = st.text_input("Search by name, phone, email")
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        query = "SELECT * FROM customers WHERE is_active = TRUE"
        params = []
        if search:
            query += " AND (full_name ILIKE %s OR phone ILIKE %s OR email ILIKE %s)"
            s = f"%{search}%"
            params.extend([s, s, s])
        query += " ORDER BY full_name"
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        if not rows:
            st.info("No customers.")
        else:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            if st.button("Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button("Download", data=csv, file_name="customers.csv", mime="text/csv")

    with tab2:
        with st.form("add_customer"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name*")
                dob = st.date_input("Date of Birth", value=None)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                phone = st.text_input("Phone")
            with col2:
                email = st.text_input("Email")
                address = st.text_area("Address")
                conditions = st.text_area("Medical Conditions (comma separated)")
                allergies = st.text_area("Allergies (comma separated)")
            submitted = st.form_submit_button("Add Customer")
            if submitted and name:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO customers (full_name, date_of_birth, gender, phone, email, address, medical_conditions, allergies)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (name, dob, gender, phone, email, address, conditions, allergies))
                conn.commit()
                cur.close()
                st.success(f"Added {name}")
                log_activity(st.session_state.user["user_id"], "Add Customer", name)
                st.cache_data.clear()
                st.experimental_rerun()
    return_db_connection(conn)

# ============================================================================
# 12. PRESCRIPTIONS
# ============================================================================

def show_prescriptions():
    st.header("📋 Prescription Management")
    require_role(["admin", "pharmacist"])
    conn = get_db_connection()

    tab1, tab2 = st.tabs(["📋 View", "➕ Add"])

    with tab1:
        search = st.text_input("Search by doctor or customer")
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        query = """
            SELECT p.*, c.full_name as customer_name
            FROM prescriptions p
            JOIN customers c ON p.customer_id = c.customer_id
        """
        params = []
        if search:
            query += " WHERE p.doctor_name ILIKE %s OR c.full_name ILIKE %s"
            s = f"%{search}%"
            params.extend([s, s])
        query += " ORDER BY p.issued_date DESC"
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        if not rows:
            st.info("No prescriptions.")
        else:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab2:
        with st.form("add_prescription"):
            customer_id = st.number_input("Customer ID", min_value=1, step=1)
            doctor = st.text_input("Doctor Name")
            issued = st.date_input("Issued Date", value=datetime.now().date())
            expiry = st.date_input("Expiry Date")
            notes = st.text_area("Notes")
            status = st.selectbox("Status", ["active", "inactive", "expired"])
            st.subheader("Prescription Items")
            med_ids = st.text_area("Medicine IDs (comma separated)")
            dosages = st.text_area("Dosages (comma separated)")
            frequencies = st.text_area("Frequencies (comma separated)")
            submitted = st.form_submit_button("Save")
            if submitted:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO prescriptions (customer_id, doctor_name, issued_date, expiry_date, notes, status)
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (customer_id, doctor, issued, expiry, notes, status))
                presc_id = cur.lastrowid
                if med_ids and dosages and frequencies:
                    mid_list = med_ids.split(',')
                    dos_list = dosages.split(',')
                    freq_list = frequencies.split(',')
                    for i in range(min(len(mid_list), len(dos_list), len(freq_list))):
                        cur.execute("""
                            INSERT INTO prescription_items (prescription_id, medicine_id, dosage, frequency)
                            VALUES (%s,%s,%s,%s)
                        """, (presc_id, int(mid_list[i].strip()), dos_list[i].strip(), freq_list[i].strip()))
                conn.commit()
                cur.close()
                st.success("Prescription saved!")
                log_activity(st.session_state.user["user_id"], "Add Prescription", f"ID {presc_id}")
                st.cache_data.clear()
                st.experimental_rerun()
    return_db_connection(conn)

# ============================================================================
# 13. SUPPLIERS
# ============================================================================

def show_suppliers():
    st.header("🏢 Supplier Management")
    require_role(["admin", "manager"])
    conn = get_db_connection()

    tab1, tab2 = st.tabs(["📋 List", "➕ Add"])
    with tab1:
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cur.execute("SELECT * FROM suppliers ORDER BY name")
        rows = cur.fetchall()
        cur.close()
        if not rows:
            st.info("No suppliers.")
        else:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab2:
        with st.form("add_supplier"):
            name = st.text_input("Supplier Name*")
            contact = st.text_input("Contact Person")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
            address = st.text_area("Address")
            tax = st.text_input("Tax ID")
            submitted = st.form_submit_button("Add")
            if submitted and name:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO suppliers (name, contact_person, phone, email, address, tax_id)
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (name, contact, phone, email, address, tax))
                conn.commit()
                cur.close()
                st.success("Added")
                st.cache_data.clear()
                st.experimental_rerun()
    return_db_connection(conn)

# ============================================================================
# 14. PURCHASE ORDERS
# ============================================================================

def show_purchase_orders():
    st.header("📋 Purchase Orders")
    require_role(["admin", "manager"])
    conn = get_db_connection()

    tab1, tab2 = st.tabs(["📋 View", "➕ Create"])

    with tab1:
        status_filter = st.selectbox("Status", ["All", "pending", "received", "cancelled"])
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        query = """
            SELECT po.*, s.name as supplier_name
            FROM purchase_orders po
            JOIN suppliers s ON po.supplier_id = s.supplier_id
        """
        if status_filter != "All":
            query += f" WHERE po.status = '{status_filter}'"
        query += " ORDER BY po.order_date DESC"
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        if not rows:
            st.info("No orders.")
        else:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            with st.expander("📦 Receive Order"):
                order_id = st.number_input("Order ID", min_value=1, step=1)
                if st.button("Receive Order"):
                    cur = conn.cursor()
                    cur.execute("UPDATE purchase_orders SET status = 'received' WHERE order_id = %s", (order_id,))
                    cur.execute("SELECT * FROM purchase_order_items WHERE order_id = %s", (order_id,))
                    items = cur.fetchall()
                    for item in items:
                        med_id = item[2]
                        qty = item[3]
                        cur.execute("UPDATE medicines SET quantity = quantity + %s WHERE medicine_id = %s", (qty, med_id))
                        cur.execute("""
                            INSERT INTO stock_movements (medicine_id, quantity_change, movement_type, reference_id, user_id)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (med_id, qty, 'purchase', order_id, st.session_state.user["user_id"]))
                    conn.commit()
                    cur.close()
                    st.success("Order received and stock updated!")
                    log_activity(st.session_state.user["user_id"], "Receive Order", f"Order {order_id}")
                    st.cache_data.clear()
                    st.experimental_rerun()

    with tab2:
        with st.form("create_purchase_order"):
            supplier = st.selectbox("Supplier", list(get_suppliers_dict().keys()), format_func=lambda x: get_suppliers_dict().get(x, x))
            order_date = st.date_input("Order Date", value=datetime.now().date())
            expected = st.date_input("Expected Delivery Date")
            notes = st.text_area("Notes")
            st.subheader("Order Items")
            med_ids = st.text_area("Medicine IDs (comma separated)")
            quantities = st.text_area("Quantities (comma separated)")
            unit_prices = st.text_area("Unit Prices (comma separated)")
            submitted = st.form_submit_button("Create Order")
            if submitted and supplier and med_ids and quantities and unit_prices:
                mid_list = [int(x.strip()) for x in med_ids.split(',')]
                qty_list = [int(x.strip()) for x in quantities.split(',')]
                price_list = [float(x.strip()) for x in unit_prices.split(',')]
                if len(mid_list) == len(qty_list) == len(price_list):
                    cur = conn.cursor()
                    total = sum(qty_list[i] * price_list[i] for i in range(len(mid_list)))
                    cur.execute("""
                        INSERT INTO purchase_orders (supplier_id, order_date, expected_delivery, status, total_amount, notes, created_by)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """, (supplier, order_date, expected, 'pending', total, notes, st.session_state.user["user_id"]))
                    order_id = cur.lastrowid
                    for i in range(len(mid_list)):
                        cur.execute("""
                            INSERT INTO purchase_order_items (order_id, medicine_id, quantity, unit_price, total_price)
                            VALUES (%s,%s,%s,%s,%s)
                        """, (order_id, mid_list[i], qty_list[i], price_list[i], qty_list[i]*price_list[i]))
                    conn.commit()
                    cur.close()
                    st.success(f"Order created with ID {order_id}")
                    log_activity(st.session_state.user["user_id"], "Create PO", f"Order {order_id}")
                    st.cache_data.clear()
                    st.experimental_rerun()
                else:
                    st.error("Lists must be the same length.")
    return_db_connection(conn)

# ============================================================================
# 15. STOCK ADJUSTMENTS
# ============================================================================

def show_stock_adjustments():
    st.header("📦 Stock Adjustments")
    require_role(["admin", "manager"])

    with st.form("adjust_stock"):
        med_id = st.number_input("Medicine ID", min_value=1, step=1)
        change = st.number_input("Quantity Change (positive for increase, negative for decrease)", step=1)
        reason = st.text_area("Reason")
        submitted = st.form_submit_button("Apply Adjustment")
        if submitted:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=extras.RealDictCursor)
            cur.execute("SELECT quantity FROM medicines WHERE medicine_id = %s", (med_id,))
            row = cur.fetchone()
            if not row:
                st.error("Medicine not found")
            else:
                new_qty = row['quantity'] + change
                if new_qty < 0:
                    st.error("Stock cannot be negative.")
                else:
                    cur2 = conn.cursor()
                    cur2.execute("UPDATE medicines SET quantity = %s WHERE medicine_id = %s", (new_qty, med_id))
                    cur2.execute("""
                        INSERT INTO stock_movements (medicine_id, quantity_change, movement_type, user_id, notes)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (med_id, change, 'adjustment', st.session_state.user["user_id"], reason))
                    conn.commit()
                    cur2.close()
                    st.success(f"Adjusted. New quantity: {new_qty}")
                    log_activity(st.session_state.user["user_id"], "Stock Adjustment", f"Medicine {med_id} change {change}")
                    st.cache_data.clear()
                    st.experimental_rerun()
            cur.close()
            return_db_connection(conn)

# ============================================================================
# 16. SALES RETURNS
# ============================================================================

def show_sales_returns():
    st.header("🔄 Sales Returns")
    require_role(["admin", "manager"])
    conn = get_db_connection()

    invoice_to_return = st.text_input("Invoice Number to Return")
    if invoice_to_return:
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cur.execute("SELECT sale_id, total_amount FROM sales WHERE invoice_number = %s", (invoice_to_return,))
        sale = cur.fetchone()
        cur.close()
        if not sale:
            st.error("Invoice not found.")
        else:
            sale_id = sale['sale_id']
            total = sale['total_amount']
            st.write(f"Total amount: ${total:.2f}")
            reason = st.text_area("Return Reason")
            if st.button("Process Return"):
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO sales_returns (sale_id, reason, total_refund, user_id)
                    VALUES (%s, %s, %s, %s)
                """, (sale_id, reason, total, st.session_state.user["user_id"]))
                return_id = cur.lastrowid
                cur.execute("SELECT medicine_id, quantity, unit_price FROM sale_items WHERE sale_id = %s", (sale_id,))
                items = cur.fetchall()
                for item in items:
                    med_id = item[0]
                    qty = item[1]
                    cur.execute("UPDATE medicines SET quantity = quantity + %s WHERE medicine_id = %s", (qty, med_id))
                    cur.execute("""
                        INSERT INTO stock_movements (medicine_id, quantity_change, movement_type, reference_id, user_id)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (med_id, qty, 'return', return_id, st.session_state.user["user_id"]))
                    cur.execute("""
                        INSERT INTO return_items (return_id, medicine_id, quantity, unit_price)
                        VALUES (%s, %s, %s, %s)
                    """, (return_id, med_id, qty, item[2]))
                conn.commit()
                cur.close()
                st.success("Return processed, stock restored.")
                log_activity(st.session_state.user["user_id"], "Sales Return", f"Invoice {invoice_to_return}")
                st.cache_data.clear()
                st.experimental_rerun()
    return_db_connection(conn)

# ============================================================================
# 17. REPORTS (with AI forecasting)
# ============================================================================

def show_reports():
    st.header("📊 Reports & Analytics")
    require_role(["admin", "manager"])
    conn = get_db_connection()

    report_type = st.selectbox(
        "Select Report",
        ["Sales Summary", "Inventory Report", "Expiry Report", "Demand Forecast (AI)", "Activity Logs"]
    )

    if report_type == "Sales Summary":
        start = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        end = st.date_input("End Date", datetime.now())
        if st.button("Generate"):
            cur = conn.cursor(cursor_factory=extras.RealDictCursor)
            cur.execute("""
                SELECT date(sale_date) as date, COUNT(*) as sales_count, SUM(total_amount) as revenue
                FROM sales
                WHERE sale_date BETWEEN %s AND %s
                GROUP BY date(sale_date)
                ORDER BY date
            """, (start, end))
            rows = cur.fetchall()
            cur.close()
            if not rows:
                st.info("No data.")
            else:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
                fig = px.bar(df, x='date', y='revenue', title='Daily Revenue')
                st.plotly_chart(fig, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", data=csv, file_name="sales_report.csv", mime="text/csv")
                if st.button("Download PDF"):
                    pdf = generate_simple_pdf(df, "Sales Report")
                    st.download_button("PDF", data=pdf, file_name="sales_report.pdf", mime="application/pdf")

    elif report_type == "Inventory Report":
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cur.execute("SELECT category, COUNT(*) as count, SUM(quantity) as total_stock, AVG(selling_price) as avg_price FROM medicines WHERE is_active=TRUE GROUP BY category")
        rows = cur.fetchall()
        cur.close()
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            fig = px.pie(df, values='count', names='category', title='Distribution')
            st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Expiry Report":
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cur.execute("SELECT name, expiry_date, quantity FROM medicines WHERE expiry_date IS NOT NULL AND is_active=TRUE ORDER BY expiry_date")
        rows = cur.fetchall()
        cur.close()
        if rows:
            df = pd.DataFrame(rows)
            df['days'] = (pd.to_datetime(df['expiry_date']) - pd.Timestamp.now()).dt.days
            st.dataframe(df, use_container_width=True)
            fig = px.histogram(df, x='days', title='Expiry Distribution (days from now)')
            st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Demand Forecast (AI)":
        st.subheader("🧠 AI Demand Forecasting")
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cur.execute("SELECT medicine_id, name FROM medicines WHERE is_active=TRUE")
        med_list = cur.fetchall()
        cur.close()
        if not med_list:
            st.warning("No medicines.")
        else:
            med_dict = {m['medicine_id']: m['name'] for m in med_list}
            chosen = st.selectbox("Select Medicine", list(med_dict.keys()), format_func=lambda x: med_dict[x])
            days = st.slider("Forecast Days", 7, 30, 7)
            if st.button("Forecast"):
                cur = conn.cursor(cursor_factory=extras.RealDictCursor)
                cur.execute("""
                    SELECT date(sale_date) as ds, SUM(si.quantity) as y
                    FROM sales s
                    JOIN sale_items si ON s.sale_id = si.sale_id
                    WHERE si.medicine_id = %s
                    GROUP BY date(sale_date)
                    ORDER BY ds
                """, (chosen,))
                hist = cur.fetchall()
                cur.close()
                if len(hist) < 5:
                    st.warning("Not enough historical data.")
                else:
                    df_hist = pd.DataFrame(hist)
                    if PROPHET_AVAILABLE:
                        model = Prophet()
                        model.fit(df_hist)
                        future = model.make_future_dataframe(periods=days)
                        forecast = model.predict(future)
                        forecast_df = forecast[['ds','yhat']].tail(days)
                        st.dataframe(forecast_df, use_container_width=True)
                        fig = model.plot(forecast)
                        st.pyplot(fig)
                    else:
                        # Linear regression fallback
                        df_hist['day_num'] = range(1, len(df_hist)+1)
                        X = df_hist[['day_num']].values
                        y = df_hist['y'].values
                        model = LinearRegression()
                        model.fit(X, y)
                        future_days = np.array(range(len(df_hist)+1, len(df_hist)+days+1)).reshape(-1,1)
                        pred = model.predict(future_days)
                        pred_dates = [(datetime.now() + timedelta(days=i)).date() for i in range(1, days+1)]
                        forecast_df = pd.DataFrame({'Date': pred_dates, 'Predicted Demand': pred})
                        st.dataframe(forecast_df, use_container_width=True)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_hist['ds'], y=df_hist['y'], mode='lines+markers', name='Historical'))
                        fig.add_trace(go.Scatter(x=pred_dates, y=pred, mode='lines+markers', name='Forecast'))
                        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Activity Logs":
        days = st.number_input("Last N days", min_value=1, value=30)
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cur.execute("""
            SELECT l.*, u.full_name
            FROM activity_logs l
            LEFT JOIN users u ON l.user_id = u.user_id
            WHERE l.timestamp >= NOW() - INTERVAL '1 day' * %s
            ORDER BY l.timestamp DESC
        """, (days,))
        rows = cur.fetchall()
        cur.close()
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="activity_log.csv", mime="text/csv")
    return_db_connection(conn)

# ============================================================================
# 18. SETTINGS (Company, Logo, Biometric, Users)
# ============================================================================

def show_settings():
    st.header("⚙️ Settings")
    require_role(["admin"])
    conn = get_db_connection()

    company = _get_company_data()
    if not company:
        st.error("Company settings not found.")
        return

    with st.expander("🏢 Company Profile", expanded=True):
        with st.form("company_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Company Name", company['company_name'])
                address = st.text_input("Address", company['address'])
                phone = st.text_input("Phone", company['phone'])
            with col2:
                email = st.text_input("Email", company['email'])
                tax = st.text_input("Tax ID", company['tax_id'])
                currency = st.text_input("Currency Symbol", company['currency_symbol'])
            footer = st.text_area("Receipt Footer", company['receipt_footer'])
            logo_file = st.file_uploader("Upload Logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
            if logo_file:
                os.makedirs("uploads", exist_ok=True)
                logo_path = f"uploads/logo_{int(time.time())}.png"
                with open(logo_path, "wb") as f:
                    f.write(logo_file.getbuffer())
                company['logo_path'] = logo_path

            stamp_valid = company.get('stamp_valid', False)
            if stamp_valid:
                st.success("✅ Digital stamp verified – company data is authentic.")
            else:
                st.error("🚨 Digital stamp invalid – data may have been tampered!")

            submitted = st.form_submit_button("Update Company")
            if submitted:
                data = {
                    'company_name': name,
                    'address': address,
                    'phone': phone,
                    'email': email,
                    'tax_id': tax,
                    'currency_symbol': currency,
                    'receipt_footer': footer,
                    'logo_path': company.get('logo_path', '')
                }
                if _update_company_data(data):
                    st.success("Updated.")
                    st.cache_data.clear()
                    st.experimental_rerun()

    with st.expander("👥 User Management"):
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cur.execute("SELECT user_id, username, role, full_name, email, is_active FROM users")
        users = cur.fetchall()
        cur.close()
        st.dataframe(pd.DataFrame(users), use_container_width=True)

        with st.form("add_user"):
            new_user = st.text_input("Username*")
            new_pass = st.text_input("Password*", type="password")
            new_role = st.selectbox("Role", ["admin", "pharmacist", "salesman", "manager"])
            new_full = st.text_input("Full Name")
            new_email = st.text_input("Email")
            if BIOMETRIC_AVAILABLE:
                st.info("Enroll face for biometric: upload a clear face image.")
                face_file = st.file_uploader("Face Image (for biometric)", type=["jpg", "png"])
            else:
                face_file = None
            is_active = st.checkbox("Active", value=True)
            if st.form_submit_button("Add User"):
                if new_user and new_pass:
                    face_encoding_json = None
                    if face_file and BIOMETRIC_AVAILABLE:
                        file_bytes = np.asarray(bytearray(face_file.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        locations = face_recognition.face_locations(rgb)
                        if locations:
                            encodings = face_recognition.face_encodings(rgb, locations)
                            if encodings:
                                face_encoding_json = json.dumps(encodings[0].tolist())
                    cur2 = conn.cursor()
                    try:
                        cur2.execute("""
                            INSERT INTO users (username, password_hash, role, full_name, email, face_encoding, is_active)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (new_user, _hash_password(new_pass), new_role, new_full, new_email, face_encoding_json, is_active))
                        conn.commit()
                        cur2.close()
                        st.success(f"User {new_user} added.")
                        log_activity(st.session_state.user["user_id"], "Add User", new_user)
                        st.cache_data.clear()
                        st.experimental_rerun()
                    except psycopg2.IntegrityError:
                        st.error("Username already exists.")

        # Deactivate user
        user_id = st.number_input("User ID to deactivate", min_value=2, step=1)
        if st.button("Deactivate User"):
            cur2 = conn.cursor()
            cur2.execute("UPDATE users SET is_active = FALSE WHERE user_id = %s", (user_id,))
            conn.commit()
            cur2.close()
            st.success(f"User {user_id} deactivated.")
            log_activity(st.session_state.user["user_id"], "Deactivate User", str(user_id))
            st.cache_data.clear()
            st.experimental_rerun()

    # Biometric toggle
    st.subheader("🔐 Biometric Authentication")
    enable_biometric = st.checkbox("Enable biometric login (face recognition)", value=st.session_state.use_biometric)
    if enable_biometric != st.session_state.use_biometric:
        st.session_state.use_biometric = enable_biometric
        st.success("Biometric setting updated. Relogin to apply.")
        st.experimental_rerun()

    return_db_connection(conn)

# ============================================================================
# 19. MAIN APP ROUTER
# ============================================================================

def main_app():
    menu = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Inventory", "Sales (POS)", "Customers",
         "Prescriptions", "Suppliers", "Purchase Orders", "Stock Adjustments",
         "Sales Returns", "Reports", "Settings"]
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"**👤 {st.session_state.user['full_name']}** ({st.session_state.user['role']})")
    if st.sidebar.button("🚪 Logout"):
        log_activity(st.session_state.user["user_id"], "Logout", "")
        st.session_state.logged_in = False
        st.session_state.user = None
        st.cache_data.clear()
        st.experimental_rerun()

    if menu == "Dashboard":
        show_dashboard()
    elif menu == "Inventory":
        show_inventory()
    elif menu == "Sales (POS)":
        show_pos()
    elif menu == "Customers":
        show_customers()
    elif menu == "Prescriptions":
        show_prescriptions()
    elif menu == "Suppliers":
        show_suppliers()
    elif menu == "Purchase Orders":
        show_purchase_orders()
    elif menu == "Stock Adjustments":
        show_stock_adjustments()
    elif menu == "Sales Returns":
        show_sales_returns()
    elif menu == "Reports":
        show_reports()
    elif menu == "Settings":
        show_settings()

# ============================================================================
# 20. RUN
# ============================================================================

if __name__ == "__main__":
    init_db()
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
        

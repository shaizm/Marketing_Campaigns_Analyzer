## 📊 Marketing Campaign Grading & AI Analysis tool

An intelligent Streamlit application that evaluates marketing campaign performance using a weighted ranking algorithm and provides automated business insights via the Gemini.

### 🚀 Key Features:
* Multi-Metric Scoring: Calculates ROI, ROAS, CPL, CPA, and Conversion Rates.
* Weighted Rank Algorithm: Assigns an objective 0–100 score based on ROI (40%), Revenue (30%), CPL (18%), and Lead Volume (12%).
* AI Insights: Generates concise, business-focused performance summaries using Gemini AI.
* Privacy-First Analysis: Includes a "Data Masking" toggle that anonymizes campaign names and scales financial figures before sending data to the LLM.
* Interactive Visualization: Dynamic tables with color gradients to identify top and bottom performers instantly.

### 🛠️ Tech Stack
1. Frontend: Streamlit
2. Data Science: Pandas, NumPy
3. AI/LLM: Google GenAI SDK (gemini-2.5-flash-preview-09-2025)

### 📋 Prerequisites
* Python 3.9+ & A Google Gemini API Key

### ⚙️ Installation & Setup
Clone the repository:
cd campaign-scoring-app

### Install dependencies:
pip install -r requirements.txt

### Configure API Key:
Create a .env file or set a Streamlit secret:
GEMINI_API_KEY=your_api_key_here

### Run the App:
streamlit run streamlit_app.py

### 📊 Data Input Format
The app expects a CSV file with the following columns:
Campaign: Name of the campaign
Total Spend: Marketing spend in currency
Total Leads: Number of leads generated
Total Sales: Number of conversions/sales
Revenue (incl. GST): Total revenue generated

### 🛡️ Data Privacy
This app prioritizes data security. By enabling Masking, the app:
Replaces campaign names with generic identifiers (e.g., Campaign 1).
Scales financial values by a non-integer factor to hide absolute numbers while preserving performance ratios for the AI's analysis.

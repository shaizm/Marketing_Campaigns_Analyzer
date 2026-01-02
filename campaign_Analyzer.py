import streamlit as st
import pandas as pd
import numpy as np
import os
import google.genai as genai

st.set_page_config(page_title="Campaign Analysing Tool", layout="wide")
st.title("📊 Campaign Grading & Analyser")
st.markdown("Upload your campaign data (CSV with columns: `Campaign`, `Total Spend`, `Total Leads`, `Total Sales`, `Revenue (incl. GST)`)")

# --- API Key Configuration ---
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
llm_enabled = False

if api_key:
    try:
        client = genai.Client(api_key=api_key)
        llm_enabled = True
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        llm_enabled = False

if not llm_enabled:
    st.warning("⚠️ **Gemini AI Analysis Disabled:** Please provide a valid `GEMINI_API_KEY` in your environment or `st.secrets` to use the AI analysis feature.")


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

    required_cols = ["Campaign", "Total Spend", "Total Leads", "Total Sales", "Revenue (incl. GST)"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        st.error(f"Missing columns in uploaded CSV: {missing}")
    else:
        
        # P/L (Profit/Loss)
        df["P/L"] = (df["Revenue (incl. GST)"] - df["Total Spend"]).round(2)
        
        # CPL (Cost per Lead) - Denominator: Total Leads
        df["CPL"] = np.where(
                            (df["Total Leads"].notna()) & (df["Total Leads"] != 0),
                            (df["Total Spend"] / df["Total Leads"]).round(2),df["Total Spend"])

        # CPA (Cost per Acquisition/Sale) - Denominator: Total Sales
        df["CPA"] = np.where(
                            (df["Total Sales"].notna()) & (df["Total Sales"] != 0),
                            (df["Total Spend"] / df["Total Sales"]).round(2),df["Total Spend"])                              
        
        # ROAS (Return on Ad Spend) - Denominator: Total Spend
        df["ROAS"] = np.where(
                            (df["Total Spend"].notna()) & (df["Total Spend"] != 0),
                            (df["Revenue (incl. GST)"] / df["Total Spend"]).round(2),
                            np.where(df["Revenue (incl. GST)"] > 0, 99.0, 0.0))

        # ROI (Return on Investment) - Denominator: Total Spend
        df["ROI"] = np.where(
                            (df["Total Spend"].notna()) & (df["Total Spend"] != 0),
                            (df["P/L"] / df["Total Spend"]).round(2),
                            np.where(df["Revenue (incl. GST)"] > 0, 99.0, 0.0))        
        
        # C-rate (Conversion Rate)
        df["C-rate"] = np.divide(df["Total Sales"], df["Total Leads"], 
                                 where=df["Total Leads"]!=0, out=np.full_like(df["Total Sales"], np.nan)).round(2)
        
        # Rev. per lead (Revenue per Lead) - Denominator: Total Leads
        df["Rev. per lead"] = np.divide(df["Revenue (incl. GST)"], df["Total Leads"], 
                                         where=df["Total Leads"]!=0, out=np.full_like(df["Revenue (incl. GST)"], np.nan)).round(2)

        
        # Metric weights (Based on our discussion)
        weights = {
            "ROI": 40,
            "Revenue (incl. GST)": 30,
            "CPL": 18,
            "Total Leads": 12
        }

        positive_metrics = {"ROI","Revenue (incl. GST)","Total Leads"}
        negative_metrics = {"CPL"} 

        # Calculate weighted ranks
        total_scores = []
        for idx, row in df.iterrows():
            total = 0.0
            for col, w in weights.items():
                if col in positive_metrics:
                    # Higher value = better rank (e.g., 1 is best)
                    rank = df[col].rank(ascending=False, method="min", na_option='bottom').iloc[idx]
                else:  # negative metric (CPL)
                    # Lower value = better rank (e.g., 1 is best)
                    rank = df[col].rank(ascending=True, method="min", na_option='bottom').iloc[idx]
                
                total += rank * w
            total_scores.append(total)

        df["Campaign Score Raw"] = total_scores

        raw_min = df["Campaign Score Raw"].min()
        raw_max = df["Campaign Score Raw"].max()
        
        if np.isclose(raw_max, raw_min):
            df["Campaign Score (0–100)"] = 100.0
        else:

            df["Campaign Score (0–100)"] = (raw_max - df["Campaign Score Raw"]) / (raw_max - raw_min) * 100
            
        df_sorted = df.sort_values("Campaign Score (0–100)", ascending=False).reset_index(drop=True)
        df_sorted = df_sorted.drop(columns=["Campaign Score Raw"])
        
        
        # --- DISPLAY RESULTS ---
        st.markdown("---")
        st.header("🏆 Final Campaign Scores")
        st.caption("Scored from 0-100 where 100 being best & 0 being worst.")
        
        # --- COLOR CODING AND INDEX DISPLAY ---
        
        numeric_cols = df_sorted.select_dtypes(include=np.number).columns.tolist()
        format_dict = {col: "{:.2f}" for col in numeric_cols}

        # Display the ranked table using explicit formatting and background gradient
        styled = df_sorted.style \
            .format(format_dict) \
            .background_gradient(subset=["Campaign Score (0–100)"], cmap="RdYlGn")
            
        st.dataframe(
            styled, 
            width='stretch'
        )
        
        # --- LLM ANALYSIS BLOCK ---
        if llm_enabled:
            st.markdown("---")
            st.header("🤖 LLM Insights ")
            
            # --- Data Extraction for Prompt ---
            num_to_analyze = min(len(df_sorted), 6) 
            analysis_df_subset = pd.concat([df_sorted.head(3), df_sorted.tail(3)])
            
            # Filter columns for clarity in the prompt
            cols_to_analyze = ["Campaign", "Campaign Score (0–100)", "ROI", "Revenue (incl. GST)", "CPL", "Total Leads"]
            analysis_df = analysis_df_subset[cols_to_analyze]
            
            # Ensure Scaled Score is also rounded for the prompt
            analysis_data_string = analysis_df.round({"Campaign Score (0–100)": 2}).to_markdown(index=False)
            
            # Get the best and worst campaign names
            best_campaign_name = df_sorted.iloc[0]["Campaign"]
            worst_campaign_name = df_sorted.iloc[-1]["Campaign"]

            # --- Prompt ---
            prompt = f"""
            Analyze the campaign performance data provided in the markdown table below.
            
            Your response must be a concise, business-focused summary.
            It should help in making decision about continuing or suspending a particular campaign.
            
            Also note that the revenue is in INR currency.
            
            Specifically, summarize the key **difference** between the top performing campaigns 
            (e.g., '{best_campaign_name}') and the bottom performing campaigns (e.g., '{worst_campaign_name}'). 
            Reference key metrics like ROI, CPL, and Revenue to support your finding.
            
            Also mention few points about the other important and note worthy campaigns.
            Data Table (Metrics are rounded to 2 decimal places):
            {analysis_data_string}
            
            Focus on the 'Campaign Score (0–100)' for overall context but do not limit to the score column alone 
            and also do not mention campaign score in the output ( e.g., 100 Campaign Score).
            
            CRITICAL INSTRUCTION: For every mention of a specific campaign name (e.g., '{best_campaign_name}'), 
            you MUST wrap it in an inline code block (e.g., `Campaign Name`) to ensure consistent highlighting.
            """

            # --- Run LLM Analysis Button ---
            if st.button("🚀 Generate AI Insights"):
                with st.spinner("Analyzing data with Gemini-2.5-flash..."):
                    try:
                        # Call the Gemini API
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt
                        )
                        
                        # --- 4. Display the Analysis ---
                        st.info("💡 **AI Analysis Complete:**")
                        st.markdown(response.text)

                    except Exception as e:
                        st.error(f"An error occurred during LLM analysis. Please check your API key and try again. Error: {e}")

        # --- END CODE MODIFICATION ---

        csv = df_sorted.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Scored Data", csv, "campaign_scores.csv", "text/csv")
        
        st.markdown(
            """
            <style>
            .stDataFrame, .stButton button {
                border-radius: 12px;
            }
            .stDataFrame th, .stDataFrame td {
                padding: 8px;
            }
            .stButton button {
                background-color: #3b82f6;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                transition: background-color 0.3s;
            }
            .stButton button:hover {
                background-color: #2563eb;
            }
            </style>
            """,
            unsafe_allow_html=True
        )


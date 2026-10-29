import streamlit as st


def apply_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        html {
            font-size: 18px !important;
        }

        [data-testid="stAppViewContainer"] {
            font-size: 1.05rem;
        }

        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1800px;
            padding: 2rem clamp(1.25rem, 3.2vw, 3.75rem) 4rem;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] button {
            font-size: 1rem !important;
        }

        h1 {
            font-size: clamp(2.2rem, 3.4vw, 3rem) !important;
            line-height: 1.15 !important;
            letter-spacing: -0.035em;
            margin-bottom: 0.35rem !important;
            white-space: normal;
            overflow-wrap: anywhere;
        }

        h2 {
            font-size: clamp(1.55rem, 2.3vw, 2rem) !important;
            line-height: 1.25 !important;
            margin-top: 1.3rem !important;
        }

        h3 {
            font-size: 1.3rem !important;
            line-height: 1.3 !important;
        }

        p,
        li,
        label,
        [data-testid="stCaptionContainer"] {
            font-size: 1rem !important;
            line-height: 1.55 !important;
        }

        [data-testid="stCaptionContainer"] {
            opacity: 0.82;
        }

        [data-testid="stMetric"] {
            min-height: 112px;
            padding: 1rem 1.15rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            background: rgba(128, 128, 128, 0.055);
        }

        [data-testid="stMetricLabel"] p {
            font-size: 0.95rem !important;
            line-height: 1.35 !important;
        }

        [data-testid="stMetricValue"] {
            font-size: clamp(1.55rem, 2vw, 2.1rem) !important;
            line-height: 1.2 !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
        }

        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-primary"],
        [data-testid="stDownloadButton"] button {
            min-height: 44px;
            padding: 0.45rem 1rem;
            border-radius: 10px;
            font-size: 1rem !important;
            font-weight: 600;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [role="combobox"],
        [data-testid="stMultiSelect"] [role="combobox"] {
            min-height: 44px;
            font-size: 1rem !important;
        }

        [data-testid="stFileUploader"] section {
            min-height: 112px;
            border-radius: 12px;
        }

        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stTabs"] button {
            min-height: 48px;
            font-size: 1rem !important;
            font-weight: 600;
        }

        @media (max-width: 900px) {
            [data-testid="stAppViewContainer"] .main .block-container {
                padding: 1.1rem 1rem 2.5rem;
            }

            h1 {
                font-size: 1.85rem !important;
                letter-spacing: -0.025em;
            }

            h2 {
                font-size: 1.45rem !important;
            }

        }
        </style>
        """,
        unsafe_allow_html=True,
    )

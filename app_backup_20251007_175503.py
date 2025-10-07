def _apply_theme(theme_choice: str) -> str:
    """
    Returns Plotly template and injects CSS for page + sidebar.
    Supports System/Dark/Light.
    """
    if theme_choice == "System":
        st.markdown("""
        <style>
        @media (prefers-color-scheme: dark) {
          html, body, [data-testid="stAppViewContainer"] { background: #0e1117; color: #e6e6e6; }
          [data-testid="stSidebar"] { background: #161a22; }
          [data-testid="stSidebar"] * { color: #e6e6e6 !important; }
          /* inputs */
          .stSelectbox, .stTextInput, .stNumberInput, .stDateInput, .stCheckbox, .stButton { color: #e6e6e6 !important; }
          input, select, textarea { color: #e6e6e6 !important; }
          label, p, span, div { color: inherit; }
        }
        @media (prefers-color-scheme: light) {
          html, body, [data-testid="stAppViewContainer"] { background: #ffffff; color: #111111; }
          [data-testid="stSidebar"] { background: #f6f6f6; }
          [data-testid="stSidebar"] * { color: #111111 !important; }
        }
        </style>
        """, unsafe_allow_html=True)
        return "plotly_dark"
    elif theme_choice == "Dark":
        st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"] { background: #0e1117; color: #e6e6e6; }
        [data-testid="stSidebar"] { background: #161a22; }
        [data-testid="stSidebar"] * { color: #e6e6e6 !important; }
        .stSelectbox, .stTextInput, .stNumberInput, .stDateInput, .stCheckbox, .stButton { color: #e6e6e6 !important; }
        input, select, textarea { color: #e6e6e6 !important; }
        label, p, span, div { color: inherit; }
        </style>
        """, unsafe_allow_html=True)
        return "plotly_dark"
    else:  # Light
        st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"] { background: #ffffff; color: #111111; }
        [data-testid="stSidebar"] { background: #f6f6f6; }
        [data-testid="stSidebar"] * { color: #111111 !important; }
        </style>
        """, unsafe_allow_html=True)
        return "plotly_white"



import streamlit as st


def show_info_card(title: str, value: str, description: str) -> None:
    st.markdown(f"**{title}**")
    st.info(f"{value}\n\n{description}")

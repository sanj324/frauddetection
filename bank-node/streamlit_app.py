# Record-Level SHAP Force Plot (safe version)
st.markdown("### 🔍 <span style='color:#aa3333;'>Record-Level SHAP Force Plot</span>", unsafe_allow_html=True)

for i in range(min(3, len(df))):
    st.markdown(f"**Record {i + 1}**")

    # Create a figure manually for matplotlib-based force plot
    fig_force, ax_force = plt.subplots(figsize=(10, 1))

    # Safely extract SHAP values
    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    shap_val = shap_values[1][i] if isinstance(shap_values, list) else shap_values[i]
    features_row = df_features_only.iloc[i]

    shap.force_plot(
        base_value=base_val,
        shap_values=shap_val,
        features=features_row,
        matplotlib=True,
        show=False
    )

    # Render the figure
    st.pyplot(fig_force)

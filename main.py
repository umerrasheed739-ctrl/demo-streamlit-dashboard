"""
Streamlit Charts & Workflow Guide
==================================
This application demonstrates how Streamlit handles data visualization, user input,
state management, and reactive execution.

Key Streamlit Workflow Principles:
1. Reactive Execution: Streamlit reruns the script from top to bottom whenever a widget is interacted with.
2. Caching: Use `@st.cache_data` for compute/loading functions so expensive data isn't recomputed on every rerun.
3. State Management: Use `st.session_state` to store values that need to persist across reruns.
4. Layout Flexibility: Use `st.sidebar`, `st.columns`, `st.tabs`, and `st.expander` to organize dashboards cleanly.

Run this script in terminal with:
    streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

# ==============================================================================
# 1. Page Configuration (Must be the first Streamlit command in the script)
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Charts & Workflow Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. State Management Demo (st.session_state)
# ==============================================================================
# Streamlit reruns the entire file on any widget interaction.
# st.session_state acts as memory across these reruns.
if "counter" not in st.session_state:
    st.session_state.counter = 0

if "dataset_version" not in st.session_state:
    st.session_state.dataset_version = 1

# ==============================================================================
# 3. Cached Data Generation (@st.cache_data)
# ==============================================================================
# Decorating with @st.cache_data ensures the data is only generated when inputs change,
# preventing unnecessary re-computations when interacting with other widgets.
@st.cache_data
def generate_sample_data(n_points: int, seed: int):
    """Generates synthetic time-series, categorical, and continuous data."""
    np.random.seed(seed)
    
    # 1. Time Series Data
    dates = pd.date_range(start="2025-01-01", periods=n_points, freq="D")
    sales_a = np.cumsum(np.random.randn(n_points) * 20 + 50) + 500
    sales_b = np.cumsum(np.random.randn(n_points) * 15 + 40) + 400
    profit = sales_a * 0.25 + np.random.randn(n_points) * 15
    
    time_df = pd.DataFrame({
        "Date": dates,
        "Product A": sales_a,
        "Product B": sales_b,
        "Profit": profit
    })
    
    # 2. Categorical / Multivariate Data
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Toys"]
    regions = ["North", "South", "East", "West"]
    
    cat_df = pd.DataFrame({
        "TransactionID": range(1, n_points + 1),
        "Category": np.random.choice(categories, size=n_points),
        "Region": np.random.choice(regions, size=n_points),
        "Rating": np.random.uniform(2.5, 5.0, size=n_points).round(1),
        "UnitsSold": np.random.randint(1, 50, size=n_points),
        "UnitCost": np.random.uniform(10.0, 150.0, size=n_points).round(2),
        "SatisfactionScore": np.random.normal(loc=78, scale=12, size=n_points).clip(0, 100)
    })
    cat_df["Revenue"] = (cat_df["UnitsSold"] * cat_df["UnitCost"]).round(2)
    
    return time_df, cat_df

# ==============================================================================
# 4. Sidebar: User Controls & Parameters
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Controls & Parameters")
    st.write("Widgets trigger an automatic script rerun when values change.")
    
    sample_size = st.slider(
        label="Number of Data Points",
        min_value=30,
        max_value=365,
        value=90,
        step=15,
        help="Adjusting this triggers the cached generator function with a new argument."
    )
    
    selected_seed = st.number_input(
        label="Random Seed",
        min_value=1,
        max_value=9999,
        value=42,
        help="Seed for reproducible pseudo-random numbers."
    )
    
    selected_categories = st.multiselect(
        label="Filter Categories",
        options=["Electronics", "Clothing", "Home & Kitchen", "Books", "Toys"],
        default=["Electronics", "Clothing", "Home & Kitchen"]
    )
    
    color_palette = st.selectbox(
        label="Seaborn Color Palette",
        options=["viridis", "mako", "rocket", "crest", "magma"]
    )
    
    st.divider()
    
    # Session state counter widget
    st.subheader("Session State Demo")
    if st.button("Click to Increment Counter"):
        st.session_state.counter += 1
    st.write(f"Button Click Count: **{st.session_state.counter}**")

# Load datasets based on sidebar inputs
time_df, cat_df = generate_sample_data(sample_size, selected_seed)

# Filter categorical data based on multiselect
filtered_cat_df = cat_df[cat_df["Category"].isin(selected_categories)]

# ==============================================================================
# 5. Main Dashboard Header & Metrics
# ==============================================================================
st.title("📊 Streamlit Data Visualization & Workflow Demo")
st.caption(
    "Explore built-in charts, Matplotlib, Seaborn, Plotly, and Altair while "
    "understanding Streamlit's reactive execution pipeline."
)

# Metric Summary Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Total Transactions",
        value=f"{len(filtered_cat_df):,}",
        delta=f"{len(filtered_cat_df) - len(cat_df)} vs full"
    )
with col2:
    total_rev = filtered_cat_df["Revenue"].sum() if not filtered_cat_df.empty else 0
    st.metric(
        label="Total Revenue",
        value=f"${total_rev:,.2f}",
        delta="+8.4% vs last period"
    )
with col3:
    avg_rating = filtered_cat_df["Rating"].mean() if not filtered_cat_df.empty else 0
    st.metric(
        label="Average Rating",
        value=f"{avg_rating:.2f} ⭐" if not filtered_cat_df.empty else "N/A"
    )
with col4:
    avg_sat = filtered_cat_df["SatisfactionScore"].mean() if not filtered_cat_df.empty else 0
    st.metric(
        label="Satisfaction Score",
        value=f"{avg_sat:.1f}/100",
        delta="+2.1"
    )

st.divider()

# ==============================================================================
# 6. Organized Visualizations Using Tabs
# ==============================================================================
tab_native, tab_plt_sns, tab_plotly, tab_altair, tab_workflow = st.tabs([
    "📈 1. Native Streamlit Charts",
    "🎨 2. Matplotlib & Seaborn",
    "✨ 3. Interactive Plotly",
    "📊 4. Altair Declarative",
    "🧠 5. Streamlit Workflow Explained"
])

# ------------------------------------------------------------------------------
# TAB 1: Built-in Native Streamlit Charts
# ------------------------------------------------------------------------------
with tab_native:
    st.subheader("Fast, Zero-Configuration Built-in Charts")
    st.markdown(
        """
        Streamlit provides native chart commands (`st.line_chart`, `st.area_chart`, 
        `st.bar_chart`, `st.scatter_chart`). Under the hood, these use Altair and are 
        optimized for quick exploratory visualization directly from pandas DataFrames.
        """
    )
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write("##### Line Chart: Cumulative Sales Trend")
        # st.line_chart accepts a DataFrame with dates or indices
        st.line_chart(
            data=time_df.set_index("Date")[["Product A", "Product B"]],
            use_container_width=True
        )
        st.caption("`st.line_chart(...)` automatically plots numeric columns over an index.")

    with col_chart2:
        st.write("##### Area Chart: Profit Progression")
        st.area_chart(
            data=time_df.set_index("Date")[["Profit"]],
            color="#2ECC71",
            use_container_width=True
        )
        st.caption("`st.area_chart(...)` fills the region under the curve.")

    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.write("##### Bar Chart: Total Revenue by Category")
        if not filtered_cat_df.empty:
            cat_revenue = filtered_cat_df.groupby("Category")["Revenue"].sum().reset_index()
            st.bar_chart(
                data=cat_revenue,
                x="Category",
                y="Revenue",
                use_container_width=True
            )
        else:
            st.warning("Please select at least one category from the sidebar.")

    with col_chart4:
        st.write("##### Scatter Chart: Unit Cost vs Satisfaction Score")
        if not filtered_cat_df.empty:
            st.scatter_chart(
                data=filtered_cat_df,
                x="UnitCost",
                y="SatisfactionScore",
                color="Category",
                size="UnitsSold",
                use_container_width=True
            )
        else:
            st.warning("Please select at least one category from the sidebar.")

# ------------------------------------------------------------------------------
# TAB 2: Matplotlib and Seaborn
# ------------------------------------------------------------------------------
with tab_plt_sns:
    st.subheader("Statistical Plots with Matplotlib & Seaborn")
    st.markdown(
        """
        To render Matplotlib or Seaborn charts, create a `fig, ax = plt.subplots()` object 
        and pass it to `st.pyplot(fig)`. 
        
        > **Best Practice:** Avoid using `plt.show()`. Explicitly create figure objects 
        > to prevent concurrency and state leakage between app runs.
        """
    )
    
    col_sns1, col_sns2 = st.columns(2)
    
    with col_sns1:
        st.write("##### Seaborn Distribution & KDE Plot")
        if not filtered_cat_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.histplot(
                data=filtered_cat_df,
                x="SatisfactionScore",
                hue="Category",
                kde=True,
                palette=color_palette,
                ax=ax,
                multiple="stack"
            )
            ax.set_title("Distribution of Customer Satisfaction Score", fontsize=12)
            ax.set_xlabel("Satisfaction Score (0 - 100)")
            ax.set_ylabel("Count")
            plt.tight_layout()
            
            # Render through Streamlit
            st.pyplot(fig)
        else:
            st.info("No categories selected.")

    with col_sns2:
        st.write("##### Correlation Heatmap (Continuous Features)")
        if not filtered_cat_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            numeric_cols = ["UnitsSold", "UnitCost", "SatisfactionScore", "Revenue", "Rating"]
            corr_matrix = filtered_cat_df[numeric_cols].corr()
            
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap=color_palette,
                cbar=True,
                ax=ax,
                linewidths=0.5
            )
            ax.set_title("Feature Correlation Matrix", fontsize=12)
            plt.tight_layout()
            
            st.pyplot(fig)
        else:
            st.info("No data available.")

    with st.expander("🔍 See Matplotlib Boxplot by Region"):
        if not filtered_cat_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(
                data=filtered_cat_df,
                x="Region",
                y="Revenue",
                hue="Category",
                palette=color_palette,
                ax=ax
            )
            ax.set_title("Revenue Dispersion by Region and Category")
            plt.tight_layout()
            st.pyplot(fig)

# ------------------------------------------------------------------------------
# TAB 3: Interactive Plotly Visualizations
# ------------------------------------------------------------------------------
with tab_plotly:
    st.subheader("Interactive Visualizations with Plotly")
    st.markdown(
        """
        Plotly charts provide built-in hovering, zooming, panning, and lasso-selection.
        Render them with `st.plotly_chart(fig, use_container_width=True)`.
        """
    )
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.write("##### Interactive Scatter: Revenue vs Satisfaction")
        if not filtered_cat_df.empty:
            fig_scatter = px.scatter(
                filtered_cat_df,
                x="Revenue",
                y="SatisfactionScore",
                color="Category",
                size="UnitsSold",
                hover_data=["Region", "Rating", "TransactionID"],
                title="Customer Satisfaction vs Transaction Revenue",
                template="plotly_white"
            )
            fig_scatter.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Select categories from the sidebar.")

    with col_p2:
        st.write("##### Sunburst Hierarchy: Region → Category → Revenue")
        if not filtered_cat_df.empty:
            fig_sunburst = px.sunburst(
                filtered_cat_df,
                path=["Region", "Category"],
                values="Revenue",
                color="Rating",
                color_continuous_scale="Viridis",
                title="Revenue Breakdown by Region and Category"
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.warning("Select categories from the sidebar.")

    # 3D Plotly Surface or Scatter
    st.write("##### 3D Interactive Scatter Plot")
    if not filtered_cat_df.empty:
        fig_3d = px.scatter_3d(
            filtered_cat_df,
            x="UnitsSold",
            y="UnitCost",
            z="SatisfactionScore",
            color="Category",
            size="Revenue",
            opacity=0.8,
            title="3D Perspective: Units Sold vs Unit Cost vs Satisfaction Score",
            height=500
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: Declarative Visualizations with Altair
# ------------------------------------------------------------------------------
with tab_altair:
    st.subheader("Declarative Grammar of Graphics with Altair")
    st.markdown(
        """
        Altair uses a concise declarative grammar. Streamlit has first-class integration 
        with Altair via `st.altair_chart(chart, use_container_width=True)`.
        """
    )
    
    if not filtered_cat_df.empty:
        # Altair chart with tooltips and interactive selection
        brush = alt.selection_interval()
        
        points = alt.Chart(filtered_cat_df).mark_circle(size=80).encode(
            x=alt.X("UnitCost:Q", title="Unit Cost ($)"),
            y=alt.Y("Revenue:Q", title="Revenue ($)"),
            color=alt.condition(brush, "Category:N", alt.value("lightgray")),
            tooltip=["TransactionID", "Category", "Region", "Revenue", "Rating"]
        ).add_params(
            brush
        ).properties(
            title="Drag to select a region (Interactive Brush Filtering)",
            height=350
        )
        
        bars = alt.Chart(filtered_cat_df).mark_bar().encode(
            x=alt.X("count()", title="Transactions Selected"),
            y=alt.Y("Category:N"),
            color="Category:N"
        ).transform_filter(
            brush
        ).properties(
            height=150
        )
        
        compound_chart = (points & bars).resolve_scale(color="independent")
        st.altair_chart(compound_chart, use_container_width=True)
    else:
        st.warning("No data available.")

# ------------------------------------------------------------------------------
# TAB 5: Streamlit Workflow & Architecture Guide
# ------------------------------------------------------------------------------
with tab_workflow:
    st.subheader("🧠 Understanding How Streamlit Works")
    
    st.markdown("""
    ### 1. Top-to-Bottom Execution Model
    - **Traditional Web Frameworks (Flask/Django/FastAPI):** Request-response cycle where you define routes, templates, and API endpoints.
    - **Streamlit Model:** The entire Python file runs as a simple script from Line 1 to the end. Every time the user interacts with a widget (slider, button, text input), **Streamlit executes the script again from scratch** with the new widget values.
    
    ### 2. Caching Mechanism (`@st.cache_data` & `@st.cache_resource`)
    Because the script reruns frequently, expensive operations (fetching APIs, querying databases, heavy ML calculations) must be cached:
    - `@st.cache_data`: Caches serializable data (DataFrames, arrays, primitives, dicts).
    - `@st.cache_resource`: Caches global stateful resources (database connections, ML models, clients).
    """)

    st.code(
        '''
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
        ''',
        language="python"
    )

    st.markdown("""
    ### 3. State Management (`st.session_state`)
    Since global variables re-initialize on every rerun, Streamlit provides a dictionary-like object called `st.session_state` that persists for each user's browser session:
    """)

    st.code(
        '''
# Initialize state if not present
if "cart" not in st.session_state:
    st.session_state.cart = []

# Update state upon button click
if st.button("Add Item"):
    st.session_state.cart.append("Item")
    st.rerun()  # Optional programmatic rerun
        ''',
        language="python"
    )

    st.markdown("""
    ### 4. Layout Primitives
    - `st.sidebar`: Collapsible left sidebar for filters and controls.
    - `st.columns`: Horizontal multi-column splits.
    - `st.tabs`: Tabbed views for decluttering dashboard sections.
    - `st.expander`: Collapsible containers for details/tables.
    - `st.container`: Logical grouping for widgets or dynamic layout swapping.
    """)

# ==============================================================================
# 7. Raw Data Preview Section
# ==============================================================================
with st.expander("📁 View Underlying Data Table"):
    st.dataframe(filtered_cat_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_cat_df.to_csv(index=False),
        file_name="filtered_data.csv",
        mime="text/csv"
    )

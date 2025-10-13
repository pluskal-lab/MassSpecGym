# app.py
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_ag_grid as dag
import plotly.graph_objects as go

# ---------------------------
# 1) Load CSVs
# ---------------------------
RESULTS_DIR = Path("results")

DF_MAP: Dict[str, pd.DataFrame] = {
    "De novo molecule generation": pd.read_csv(RESULTS_DIR / "de_novo.csv"),
    "De novo molecule generation (bonus)": pd.read_csv(RESULTS_DIR / "de_novo_bonus.csv"),
    "Molecule retrieval": pd.read_csv(RESULTS_DIR / "retrieval.csv"),
    "Molecule retrieval (bonus)": pd.read_csv(RESULTS_DIR / "retrieval_bonus.csv"),
    "Spectrum simulation": pd.read_csv(RESULTS_DIR / "simulation.csv"),
    "Spectrum simulation (bonus)": pd.read_csv(RESULTS_DIR / "simulation_bonus.csv"),
}

def find_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "publication" in c.lower() and "date" in c.lower():
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    raise ValueError("No 'Publication Date' column found in one of the CSVs.")

# Ensure Publication Date is datetime
for name, df in DF_MAP.items():
    date_col = find_date_col(df)
    DF_MAP[name] = df.copy()
    DF_MAP[name][date_col] = pd.to_datetime(DF_MAP[name][date_col], errors="coerce")

# ---------------------------
# 2) Metric discovery
# ---------------------------
def list_metrics(df: pd.DataFrame) -> List[str]:
    """Return numeric metric columns (exclude Method, date, CI columns, and metadata)."""
    date_col = find_date_col(df)
    # Columns to exclude from metrics
    exclude_cols = {"Method", date_col, "Paper", "DOI", "Comment"}
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if " CI Low" in c or " CI High" in c:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def has_ci(df_like: pd.DataFrame, metric: str) -> bool:
    return (f"{metric} CI Low" in df_like.columns) and (f"{metric} CI High" in df_like.columns)

METRICS_BY_BENCH: Dict[str, List[str]] = {
    bench: list_metrics(df) for bench, df in DF_MAP.items()
}

# ---------------------------
# Sorting priorities per benchmark
# (col, ascending) — ascending=True means smaller is better (e.g., MCES)
# Only columns that actually exist are applied.
# ---------------------------
from typing import Optional
SORT_SPECS: Dict[str, List[Tuple[str, bool]]] = {
    "De novo molecule generation": [
        ("Top-1 Tanimoto", False),
        ("Top-10 Tanimoto", False),
        ("Top-1 MCES", True),
        ("Top-10 MCES", True),
        ("Top-1 Accuracy", False),
        ("Top-10 Accuracy", False),
    ],
    "De novo molecule generation (bonus)": [
        ("Top-1 Tanimoto", False),
        ("Top-10 Tanimoto", False),
        ("Top-1 MCES", True),
        ("Top-10 MCES", True),
        ("Top-1 Accuracy", False),
        ("Top-10 Accuracy", False),
    ],
    "Molecule retrieval": [
        ("Hit Rate @ 1", False),
        ("Hit Rate @ 5", False),
        ("Hit Rate @ 20", False),
        ("MCES @ 1", True),  # adjust if your retrieval MCES column names differ
    ],
    "Molecule retrieval (bonus)": [
        ("Hit Rate @ 1", False),
        ("Hit Rate @ 5", False),
        ("Hit Rate @ 20", False),
    ],
    "Spectrum simulation": [
        ("Hit Rate @ 1", False),
        ("Hit Rate @ 5", False),
        ("Hit Rate @ 20", False),
        ("Cosine Similarity", False),
        ("Jensen-Shannon Similarity", False),
    ],
    "Spectrum simulation (bonus)": [
        ("Hit Rate @ 1", False),
        ("Hit Rate @ 5", False),
        ("Hit Rate @ 20", False),
    ],
}

def _fmt_val(v) -> str:
    """Compact numeric formatting; gracefully handle strings/missing."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return ""
    if abs(fv) < 0.1:
        return f"{fv:.4f}".rstrip("0").rstrip(".")
    return f"{fv:.2f}".rstrip("0").rstrip(".")

def add_arrow_to_metric(metric: str, bench: str) -> str:
    """Add arrow indicator to metric name showing if higher (↑) or lower (↓) is better."""
    # Check if this metric should be minimized (lower is better)
    # First check SORT_SPECS for explicit specification
    minimize = any(col == metric and asc for col, asc in SORT_SPECS.get(bench, []))
    
    # If not in SORT_SPECS, infer from metric name
    # Metrics with these keywords typically mean lower is better
    if not any(col == metric for col, _ in SORT_SPECS.get(bench, [])):
        lower_is_better_keywords = ["MCES", "mces", "Error", "error", "Loss", "loss", "Distance", "distance"]
        minimize = any(keyword in metric for keyword in lower_is_better_keywords)
    
    if minimize:
        return f"{metric} (↓)"
    else:
        return f"{metric} (↑)"

def build_display_table(df: pd.DataFrame, bench: str) -> Tuple[List[dict], List[dict]]:
    """
    Build a compact table for AG Grid with CI in brackets and proper sorting.
    Highlights best (bold) and second-best (underlined) values in each column.
    """
    date_col = find_date_col(df)
    df_work = df.copy()

    # Apply sorting priorities (only for columns that exist and are numeric)
    sort_cols: List[str] = []
    ascending_flags: List[bool] = []
    for col, asc in SORT_SPECS.get(bench, []):
        if col in df_work.columns:
            # ensure numeric for sorting; if non-numeric, coerce to NaN
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
            sort_cols.append(col)
            ascending_flags.append(asc)
    if sort_cols:
        df_work = df_work.sort_values(sort_cols, ascending=ascending_flags, na_position="last")
    else:
        # fallback: sort by date descending
        df_work = df_work.sort_values(date_col, ascending=False)

    # Build display columns per metric
    all_metrics = list_metrics(df_work)
    
    # Order metrics according to SORT_SPECS priority for this benchmark
    # This ensures consistent column ordering across all benchmarks
    sort_order = [col for col, _ in SORT_SPECS.get(bench, [])]
    # Put priority metrics first, then any remaining metrics
    metrics = [m for m in sort_order if m in all_metrics]
    metrics.extend([m for m in all_metrics if m not in metrics])
    
    disp = pd.DataFrame()
    
    # Create Method column with paper icons using markdown
    def create_method_with_icon(method, doi):
        if pd.isna(doi) or str(doi) == 'nan' or str(doi) == '':
            return method
        return f'[📄]({doi}) {method}'
    
    disp["Method"] = [create_method_with_icon(method, doi) for method, doi in zip(df_work["Method"].astype(str), df_work.get("DOI", [""] * len(df_work)))]
    
    column_defs = [{
        "field": "Method", 
        "colId": "Method", 
        "pinned": "left", 
        "width": 200,
        "cellRenderer": "markdown"
    }]

    # Calculate best and second-best for each metric
    for m in metrics:
        mean = pd.to_numeric(df_work[m], errors="coerce")
        
        # Store the numeric value for sorting
        disp[m] = mean
        
        # Determine if lower is better or higher is better
        minimize = any(col == m and asc for col, asc in SORT_SPECS.get(bench, []))
        
        # Find best and second-best values
        valid_values = mean.dropna().sort_values(ascending=minimize)
        best_val = valid_values.iloc[0] if len(valid_values) > 0 else None
        second_best_val = valid_values.iloc[1] if len(valid_values) > 1 else None
        
        # Store ranking info as integers
        def get_rank(x):
            if pd.isna(x):
                return 0
            if pd.notna(best_val) and abs(x - best_val) < 1e-10:
                return 1
            if pd.notna(second_best_val) and abs(x - second_best_val) < 1e-10:
                return 2
            return 0
        
        disp[f"{m}_rank"] = mean.apply(get_rank).astype(int)
        
        # Check if CI data exists - store in separate fields
        if has_ci(df_work, m):
            lo = pd.to_numeric(df_work[f"{m} CI Low"], errors="coerce")
            hi = pd.to_numeric(df_work[f"{m} CI High"], errors="coerce")
            
            # Store CI values (convert NaN to None for JavaScript)
            disp[f"{m}_ci_low"] = lo.where(pd.notna(lo), None)
            disp[f"{m}_ci_high"] = hi.where(pd.notna(hi), None)
            
            # Use custom value formatter to display "value (low–high)"
            column_defs.append({
                "field": m,
                "colId": m,
                "headerName": add_arrow_to_metric(m, bench),
                "sortable": True,
                "filter": True,
                "type": "numericColumn",
                "width": 180,
                "valueFormatter": {
                    "function": f"""
                    function(params) {{
                        if (params.value == null || isNaN(params.value)) return '';
                        const val = params.value.toFixed(4).replace(/\\.?0+$/, '');
                        const low = params.data['{m}_ci_low'];
                        const high = params.data['{m}_ci_high'];
                        if (low != null && high != null && !isNaN(low) && !isNaN(high)) {{
                            const lowStr = low.toFixed(4).replace(/\\.?0+$/, '');
                            const highStr = high.toFixed(4).replace(/\\.?0+$/, '');
                            return val + ' (' + lowStr + '–' + highStr + ')';
                        }}
                        return val;
                    }}
                    """
                },
                "cellClassRules": {
                    "best-value": {"function": f"params.data['{m}_rank'] === 1"},
                    "second-best-value": {"function": f"params.data['{m}_rank'] === 2"}
                }
            })
        else:
            # No CI - just numeric value
            column_defs.append({
                "field": m,
                "colId": m,
                "headerName": add_arrow_to_metric(m, bench),
                "sortable": True,
                "filter": True,
                "type": "numericColumn",
                "width": 150,
                "valueFormatter": {
                    "function": """
                    function(params) {
                        if (params.value == null || isNaN(params.value)) return '';
                        return params.value.toFixed(4).replace(/\\.?0+$/, '');
                    }
                    """
                },
                "cellClassRules": {
                    "best-value": {"function": f"params.data['{m}_rank'] === 1"},
                    "second-best-value": {"function": f"params.data['{m}_rank'] === 2"}
                }
            })

    # Publication date
    disp[date_col] = pd.to_datetime(df_work[date_col]).dt.strftime("%d %b %Y")
    column_defs.append({"field": date_col, "colId": date_col, "headerName": date_col, "sortable": True, "width": 150})
    
    # Add Paper column if it exists (truncated with ellipsis)
    if "Paper" in df_work.columns:
        disp["Paper"] = df_work["Paper"].astype(str)
        column_defs.append({
            "field": "Paper",
            "colId": "Paper",
            "headerName": "Paper",
            "sortable": True,
            "width": 300,
            "tooltipField": "Paper",  # Show full text on hover
            "cellClass": "paper-cell"  # Custom class for ellipsis styling
        })
    
    # Add DOI column with clickable links if it exists (last column)
    if "DOI" in df_work.columns:
        # Format DOI as markdown link for the markdown renderer
        def format_doi_markdown(doi_url):
            if pd.isna(doi_url) or str(doi_url) == 'nan' or str(doi_url) == '':
                return ''
            doi_str = str(doi_url)
            # Just show the doi number part
            doi_text = doi_str.replace('https://doi.org/', '')
            return f'[{doi_text}]({doi_str})'
        
        disp["DOI"] = df_work["DOI"].apply(format_doi_markdown)
        
        column_defs.append({
            "field": "DOI",
            "colId": "DOI",
            "headerName": "DOI",
            "sortable": True,
            "width": 280,
            "cellClass": "doi-cell",
            "cellRenderer": "markdown"
        })
    
    # Ensure column_defs and data are in sync
    # Extract the ordered column names from column_defs
    ordered_columns = [col_def["field"] for col_def in column_defs]
    
    # Get hidden columns that aren't in column_defs (like _rank, _ci_low, _ci_high)
    hidden_columns = [col for col in disp.columns if col not in ordered_columns]
    
    # Create a new DataFrame with columns in the exact order we want
    ordered_data = {}
    for col in ordered_columns:
        if col in disp.columns:
            ordered_data[col] = disp[col]
    
    # Add hidden columns at the end
    for col in hidden_columns:
        ordered_data[col] = disp[col]
    
    # Create new DataFrame with exact column order
    disp_ordered = pd.DataFrame(ordered_data)
    
    # Convert to records - dict order is preserved in Python 3.7+
    data = disp_ordered.to_dict("records")

    return column_defs, data

# ---------------------------
# 3) Figure builder
# ---------------------------
def build_figure(bench: str, metric: str) -> go.Figure:
    df = DF_MAP[bench]
    date_col = find_date_col(df)
    
    # Ensure date column is datetime type
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df_sorted = df.sort_values(date_col)

    fig = go.Figure()

    # --- Compute improving frontier (draw first so it's behind) ---
    # Check if metric should be minimized based on SORT_SPECS
    minimize = any(col == metric and asc for col, asc in SORT_SPECS.get(bench, []))
    
    # Build a list of progressively improving points
    df_clean = df_sorted.dropna(subset=[metric]).copy()
    improving_points = []
    
    if not df_clean.empty:
        best_so_far = None
        last_date = None
        
        # Group by date and get best method per date
        for date, group in df_clean.groupby(date_col):
            # Find best value for this date
            if minimize:
                best_idx = group[metric].idxmin()
            else:
                best_idx = group[metric].idxmax()
            
            best_row = group.loc[best_idx]
            current_value = best_row[metric]
            
            # Check if this is an improvement over previous dates
            if best_so_far is None:
                is_improvement = True
                best_so_far = current_value
            else:
                if minimize:
                    is_improvement = current_value < best_so_far
                else:
                    is_improvement = current_value > best_so_far
                
                if is_improvement:
                    best_so_far = current_value
            
            # Only add if it's an improvement and not the same date as last point
            if is_improvement:
                improving_points.append({
                    'date': date,
                    'value': current_value,
                    'method': best_row['Method']
                })
                last_date = date
        
        if improving_points:
            improving_df = pd.DataFrame(improving_points)
            fig.add_trace(
                go.Scatter(
                    x=improving_df['date'],
                    y=improving_df['value'],
                    mode="lines",
                    line=dict(color="black", width=1),  # black guide line
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # --- Plot all methods (on top, with bigger points) ---
    for method, dfm in df_sorted.groupby("Method", dropna=False):
        if dfm[metric].isna().all():
            continue
        dfm = dfm.dropna(subset=[metric])

        err = None
        if has_ci(dfm, metric):
            ci_high = pd.to_numeric(dfm.get(f"{metric} CI High"), errors="coerce")
            ci_low  = pd.to_numeric(dfm.get(f"{metric} CI Low"),  errors="coerce")
            yvals   = pd.to_numeric(dfm[metric],                  errors="coerce")
            valid = ~(ci_high.isna() | ci_low.isna() | yvals.isna())
            if valid.any():
                err = dict(
                    type="data",
                    symmetric=False,
                    array=(ci_high[valid] - yvals[valid]).tolist(),
                    arrayminus=(yvals[valid] - ci_low[valid]).tolist(),
                )

        fig.add_trace(
            go.Scatter(
                x=dfm[date_col],
                y=dfm[metric],
                mode="markers+lines",
                name=str(method),
                marker=dict(size=11, symbol="x"),
                error_y=err,
                hovertemplate=(
                    f"<b>{method}</b><br>"
                    + "%{x|%d %b %Y}<br>"
                    + f"{metric}: %{{y:.4f}}"
                    + "<br>CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]"
                    + "<extra></extra>"
                ) if has_ci(dfm, metric) and not dfm[[f"{metric} CI Low", f"{metric} CI High"]].isna().all().all() else (
                    f"<b>{method}</b><br>"
                    + "%{x|%d %b %Y}<br>"
                    + f"{metric}: %{{y:.4f}}<extra></extra>"
                ),
                customdata=dfm[[f"{metric} CI Low", f"{metric} CI High"]].values if has_ci(dfm, metric) and not dfm[[f"{metric} CI Low", f"{metric} CI High"]].isna().all().all() else None,
            )
        )

    # --- Add margins on x-axis (10% padding) ---
    xmin = df_sorted[date_col].min()
    xmax = df_sorted[date_col].max()
    if pd.notna(xmin) and pd.notna(xmax) and xmin != xmax:
        pad = (xmax - xmin) * 0.10
        x_range = [xmin - pad, xmax + pad]
    else:
        x_range = [xmin - pd.Timedelta(days=14), xmax + pd.Timedelta(days=14)]

    fig.update_layout(
        template="plotly_white",
        xaxis_title=date_col,
        yaxis_title=add_arrow_to_metric(metric, bench),
        xaxis=dict(type="date", tickformat="%d %b %Y", range=x_range),
        legend=dict(
            title="Method",
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(family="Work Sans, sans-serif", size=12),
        ),
        margin=dict(l=40, r=20, t=20, b=90),
        font=dict(family="Work Sans, sans-serif", size=14, color="#222"),
    )
    return fig

# ---------------------------
# 4) Dash app (white theme, Work Sans font, max-width container)
# ---------------------------
app = dash.Dash(__name__, external_stylesheets=[
    "https://unpkg.com/ag-grid-community/styles/ag-grid.css",
    "https://unpkg.com/ag-grid-community/styles/ag-theme-alpine.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
])
app.title = "MassSpecGym leaderboard"
server = app.server

# Add custom CSS for table styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Hover effect for icon links */
            a.icon-link:hover {
                text-decoration: underline !important;
                opacity: 0.8;
            }
            .best-value {
                font-weight: bold !important;
            }
            .second-best-value {
                text-decoration: underline !important;
            }
            .ag-theme-alpine a {
                color: #0066cc;
                text-decoration: none;
                cursor: pointer;
                pointer-events: auto;
            }
            .ag-theme-alpine a:hover {
                text-decoration: underline;
            }
            .ag-theme-alpine .ag-cell {
                line-height: 42px;
            }
            .ag-theme-alpine .ag-row {
                border-bottom: 1px solid #e0e0e0;
            }
            /* Ellipsis only for Paper and DOI columns */
            .ag-theme-alpine .paper-cell,
            .ag-theme-alpine .doi-cell {
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

DEFAULT_BENCH = "De novo molecule generation (bonus)"
DEFAULT_METRIC = "Top-1 Tanimoto"

MAX_WIDTH = "1000px"

app.layout = html.Div(
    style={
        "fontFamily": "Work Sans, sans-serif",
        "backgroundColor": "white",
        "color": "#222",
        "minHeight": "100vh",
        "padding": "18px 0",
    },
    children=[
        html.Div(
            style={
                "maxWidth": MAX_WIDTH,
                "margin": "0 auto",
                "padding": "0 20px",
                "boxSizing": "border-box",
                "width": "100%",
            },
            children=[
                html.H2(
                    "MassSpecGym leaderboard",
                    style={
                        "textAlign": "center",
                        "marginTop": 0,
                        "marginBottom": "16px",
                        "fontFamily": "Work Sans, sans-serif",
                        "color": "#222",
                    },
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "gap": "24px",
                        "marginBottom": "20px",
                    },
                    children=[
                        html.A(
                            [html.I(className="fas fa-file-alt"), " Paper"],
                            href="https://doi.org/10.48550/arXiv.2410.23326",
                            target="_blank",
                            className="icon-link",
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "6px",
                                "textDecoration": "none",
                                "color": "#0066cc",
                                "fontSize": "16px",
                                "fontWeight": "500",
                            },
                        ),
                        html.A(
                            [html.I(className="fas fa-database"), " Hugging Face"],
                            href="https://huggingface.co/datasets/roman-bushuiev/MassSpecGym",
                            target="_blank",
                            className="icon-link",
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "6px",
                                "textDecoration": "none",
                                "color": "#0066cc",
                                "fontSize": "16px",
                                "fontWeight": "500",
                            },
                        ),
                        html.A(
                            [html.I(className="fab fa-github"), " GitHub"],
                            href="https://github.com/pluskal-lab/MassSpecGym",
                            target="_blank",
                            className="icon-link",
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "6px",
                                "textDecoration": "none",
                                "color": "#0066cc",
                                "fontSize": "16px",
                                "fontWeight": "500",
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "width": "100%",
                        "marginBottom": "24px",
                    },
                    children=[
                        html.P(
                            [
                                "MassSpecGym benchmarks machine learning models for discovering molecules from tandem mass spectrometry data. ",
                                "It evaluates models on three core tasks: ",
                                html.B("molecule retrieval"),
                                " (identifying molecules from spectra), ",
                                html.B("de novo molecule generation"),
                                " (generating molecular structures from spectra), and ",
                                html.B("spectrum simulation"),
                                " (predicting spectra from molecular structures). ",
                                "Each task has a bonus challenge which assumes knowledge of the chemical formulae. ",
                                "This leaderboard tracks the state-of-the-art performance across these tasks.",
                            ],
                            style={
                                "textAlign": "justify",
                                "fontSize": "14px",
                                "lineHeight": "1.7",
                                "color": "#333",
                                "marginBottom": "0",
                                "marginTop": "0",
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "gap": "16px",
                        "flexWrap": "wrap",
                        "justifyContent": "space-between",
                        "alignItems": "flex-end",
                        "marginBottom": "10px",
                    },
                    children=[
                        html.Div(
                            style={"flex": "1", "minWidth": "300px"},
                            children=[
                                html.Label("Benchmark", style={"display": "block", "marginBottom": "4px", "fontWeight": "500"}),
                                dcc.Dropdown(
                                    id="bench-dd",
                                    options=[{"label": k, "value": k} for k in DF_MAP.keys()],
                                    value=DEFAULT_BENCH,
                                    clearable=False,
                                    style={"width": "100%"},
                                ),
                            ]
                        ),
                        html.Div(
                            style={"flex": "1", "minWidth": "300px"},
                            children=[
                                html.Label("Metric", style={"display": "block", "marginBottom": "4px", "fontWeight": "500"}),
                                dcc.Dropdown(
                                    id="metric-dd",
                                    value=DEFAULT_METRIC,
                                    clearable=False,
                                    style={"width": "100%"},
                                ),
                            ]
                        ),
                    ],
                ),
                dcc.Graph(
                    id="bench-fig",
                    figure=build_figure(DEFAULT_BENCH, DEFAULT_METRIC),
                    style={"height": "520px", "width": "100%"},
                ),
                html.Div(
                    id="table-wrap",
                    style={"marginTop": "12px"},
                    children=[
                        dag.AgGrid(
                            id="results-table",
                            columnDefs=[],
                            rowData=[],
                            columnSize="sizeToFit",
                            defaultColDef={
                                "resizable": True,
                                "sortable": True,
                                "filter": False,
                            },
                            dangerously_allow_code=True,
                            dashGridOptions={
                                "pagination": True,
                                "paginationPageSize": 20,
                                "domLayout": "autoHeight",
                                "suppressHorizontalScroll": False,
                                "rowHeight": 42,
                                "tooltipShowDelay": 500,
                                "suppressSanitizeHtml": True,
                                "suppressHtmlEncode": True,
                                "suppressColumnVirtualisation": True,
                                "suppressFieldDotNotation": True,
                                "enableBrowserTooltips": True,
                                "allowContextMenuWithControlKey": True,
                                "onGridReady": {
                                    "function": """
                                    function(params) {
                                        // Set column order explicitly
                                        const columnDefs = params.api.getColumnDefs();
                                        const columnIds = columnDefs.map(def => def.colId || def.field);
                                        params.columnApi.setColumnOrder(columnIds);
                                    }
                                    """
                                }
                            },
                            style={"height": "auto", "width": "100%"},
                            className="ag-theme-alpine",
                        ),
                    ],
                ),
            ],
        )
    ],
)

# --- Callbacks ---
@callback(
    Output("metric-dd", "options"),
    Output("metric-dd", "value"),
    Input("bench-dd", "value"),
    Input("metric-dd", "value"),
)
def update_metric_options(bench: str, current_metric: str):
    metrics = METRICS_BY_BENCH[bench]
    options = [{"label": add_arrow_to_metric(m, bench), "value": m} for m in metrics]
    if current_metric in metrics:
        return options, current_metric
    return options, (metrics[0] if metrics else None)

@callback(
    Output("bench-fig", "figure"),
    Output("results-table", "columnDefs"),
    Output("results-table", "rowData"),
    Input("bench-dd", "value"),
    Input("metric-dd", "value"),
)
def update_figure_and_table(bench: str, metric: str):
    if not bench or not metric:
        return go.Figure(template="plotly_white"), [], []
    fig = build_figure(bench, metric)
    col_defs, data = build_display_table(DF_MAP[bench], bench)
    return fig, col_defs, data

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=True)
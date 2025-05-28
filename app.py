import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from datetime import datetime
from datetime import timedelta

# ----------- CARGA DE DATOS -----------------
st.set_page_config(layout="wide", page_title="Stakeholder Timeline Dashboard")

st.title("Stakeholder Timeline & Strategic Focus - Capillar IT")

uploaded = st.sidebar.file_uploader("Carga archivo Csv", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.warning("Por favor, sube tu archivo Cleaned_Timeline_Data.csv para comenzar.")
    st.stop()

# Preprocesar fechas
df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
df['Point Date'] = df['End Date'].combine_first(df['Start Date']).dt.normalize()
df['Month'] = df['Point Date'].dt.to_period('M').dt.to_timestamp()
months_sorted = sorted(df['Month'].unique())
months_labels = [d.strftime('%b %Y') for d in months_sorted]

# ----------- MAPEOS COHERENTES -----------------
# Stakeholder roles
stakeholder_roles = ['Usuario', 'Prestador', 'Proveedor', 'Facilitador', 'Prescriptor']

def macroFase(phase):
    phase = (phase or "").lower()
    desarrollo_kw = [
        "development", "desarrollo", "service dev", "connector", "demo",
        "baselines", "module", "setup"
    ]
    implement_kw = [
        "implementation", "interoperability", "mapping", "pivoting",
        "refinement", "update", "integration", "ontology", "service integration"
    ]
    adopcion_kw = [
        "market", "approach", "adoption", "lead", "profiling", "alignment",
        "trial", "interest", "test", "readiness", "strategy", "willingness",
        "engagement", "client", "policy", "testing", "onboarding"
    ]
    if any(k in phase for k in desarrollo_kw):
        return "Desarrollo"
    if any(k in phase for k in implement_kw):
        return "Implementación"
    if any(k in phase for k in adopcion_kw):
        return "Adopción"
    return "Desarrollo"

def mapStakeholder(row, custom_weights=None):
    v = [0,0,0,0,0]
    if row['Swimlane'] == "User engagement":
        v[0] += 1; v[4] += 1; v[3] += 0.6
    if row['Swimlane'] == "Logistics":
        v[2] += 1; v[3] += 0.8
    if row['Swimlane'] == "Data economy":
        v[1] += 1; v[2] += 0.7
    if pd.notnull(row['Events']) and "barrier" in str(row['Events']).lower():
        v[3] += 0.6; v[2] += 0.5
    if pd.notnull(row['Events']) and "target" in str(row['Events']).lower():
        v[0] += 0.7; v[4] += 0.5
    if pd.notnull(row['Events']) and "milestone" in str(row['Events']).lower():
        v = [x+0.3 for x in v]
    maxv = max(v + [1])
    v = [x/maxv for x in v]
    if custom_weights is not None:
        v = custom_weights
    return v

# --- NUEVO: lista de stakeholders principales ---

stakeholder_entities = [
    'Mayorista sin dist',
    'Mayorista con dist',
    'Agrupacion mayoristas',
    'Admin mercado',
    'Admin publica',
    'Cliente mayorista',
    'Consultor externo'
]

# ----------- INTERFAZ DE USUARIO -------------
with st.sidebar:
    st.subheader("Filtrar por mes/fase")
    months_labels_all = ['All'] + months_labels
    months_sorted_all = [None] + months_sorted
    mes_idx = st.radio(
        "Elige un mes para ver los eventos y focos estratégicos:",
        options=list(range(len(months_labels_all))),
        format_func=lambda i: months_labels_all[i]
    )
    filtro_mes = months_sorted_all[mes_idx]
    selected_month_label = months_labels_all[mes_idx]

if filtro_mes is None:
    df_mes = df.copy()
else:
    df_mes = df[df['Month'] == filtro_mes].copy()

color_map = {
    'Target': '#2563eb',
    'Milestone': '#22c55e',
    'Barrier': '#ef4444'
}
rgba_map = {
    'Target': 'rgba(37,99,235,0.68)',
    'Milestone': 'rgba(34,197,94,0.68)',
    'Barrier': 'rgba(239,68,68,0.68)'
}
swimlanes = ['Logistics','User engagement', 'Data economy']

# MÁS ESPACIO entre swimlanes y phases
SWIMLANE_SPACING = 10
PHASE_SLOT = 1

y_pos = {lane: i * SWIMLANE_SPACING for i, lane in enumerate(swimlanes)}
phase_ypos = dict()

fig = go.Figure()

# 1. Líneas base (swimlane)
for lane, y in y_pos.items():
    fig.add_trace(go.Scatter(
        x=[df_mes['Start Date'].min(), df_mes['End Date'].max()],
        y=[y, y],
        mode='lines',
        line=dict(color='#CBD5E1', width=1.7, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

# 2. Phases como columnas
for lane in swimlanes:
    sub = df_mes[df_mes['Swimlane'] == lane].copy()
    if sub.empty: continue
    phases = []
    for phase, phase_df in sub.groupby('Phase'):
        start = pd.to_datetime(phase_df['Start Date'], errors='coerce').min()
        end = pd.to_datetime(phase_df['End Date'], errors='coerce').max()
        phases.append({'name': phase, 'start': start, 'end': end, 'rows': phase_df})
    phases = sorted(phases, key=lambda p: p['start'])

    slot_used = []
    for p in phases:
        slot_found = False
        for slot_idx, slot_end in enumerate(slot_used):
            if slot_end < p['start']:
                p['slot'] = slot_idx
                slot_used[slot_idx] = p['end']
                slot_found = True
                break
        if not slot_found:
            p['slot'] = len(slot_used)
            slot_used.append(p['end'])
        p['dir'] = 1 if p['slot'] % 2 == 0 else -1
        phase_ypos[(lane, p['name'])] = y_pos[lane] - PHASE_SLOT * p['slot'] * p['dir']

    for p in phases:
        ypos = phase_ypos[(lane, p['name'])]
        # Línea Phase principal
        fig.add_trace(go.Scatter(
            x=[p['start'], p['end']],
            y=[ypos, ypos],
            mode='lines',
            line=dict(color='#0f172a', width=6, shape='spline'),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Sombra de Phase
        fig.add_trace(go.Scatter(
            x=[p['start'], p['end']],
            y=[ypos-0.09, ypos-0.09],
            mode='lines',
            line=dict(color='#94a3b8', width=13, dash='dot', shape='spline'),
            showlegend=False,
            hoverinfo='skip',
            opacity=0.13
        ))
        swimlane_colors = {
            'User engagement': "#60a5fa",   # Azul
            'Logistics': "#f59e42",         # Naranja
            'Data economy': "#43aa8b"       # Verde
        }

        # Barra phase más grueso
        BAR_HEIGHT = 0.33
        color_swimlane = swimlane_colors.get(lane, "#60a5fa")  # Azul por defecto si no está
        fig.add_trace(go.Scatter(
            x=[p['start'], p['end'], p['end'], p['start'], p['start']],
            y=[ypos-BAR_HEIGHT, ypos-BAR_HEIGHT, ypos+BAR_HEIGHT, ypos+BAR_HEIGHT, ypos-BAR_HEIGHT],
            fill="toself",
            fillcolor=color_swimlane,
            line=dict(width=3, color="#000000"),
            mode='lines',
            showlegend=False,
            hoverinfo='skip'
        ))

        # Círculo con la fecha si End Date no es día 1
        if p['end'].day != 1:
            fig.add_trace(go.Scatter(
                x=[p['end']],
                y=[ypos],
                mode='markers+text',
                marker=dict(
                    size=38,
                    color='#1e293b',
                    line=dict(width=3, color='#fff')
                ),
                text=[p['end'].strftime('%d %b')],
                textfont=dict(
                    size=10,
                    color='#fff',
                    family="Montserrat, Arial"
                ),
                textposition="middle center",
                showlegend=False,
                hoverinfo='skip'
            ))

        bar_width_seconds = (p['end'] - p['start']).total_seconds()
        text = p['name']
        char_count = len(text)
        x_unit_to_px = 28
        font_size_max = 13
        text_width_est = char_count * font_size_max * 0.6
        bar_width_px = bar_width_seconds * x_unit_to_px
        if text_width_est > bar_width_px:
            words = text.split()
            if len(words) > 1:
                mid = len(words) // 2
                line1 = ' '.join(words[:mid])
                line2 = ' '.join(words[mid:])
                phase_label = f"<b>{line1}<br>{line2}</b>"
            else:
                phase_label = f"<b>{text}</b>"
            font_size = max(8, int(bar_width_px / (char_count * 0.6)))
        else:
            phase_label = f"<b>{text}</b>"
            font_size = font_size_max

        xtext = p['start'] + (p['end'] - p['start']) / 2
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,-1), (1,1), (-1,1), (1,-1)]:
            fig.add_annotation(
                x=xtext,
                y=ypos + dy * 0.012,
                text=phase_label,
                showarrow=False,
                font=dict(size=font_size, color="#000", family="Montserrat, Arial"),
                xanchor="center", yanchor="middle",
                align="center",
                opacity=1,
            )
        fig.add_annotation(
            x=xtext,
            y=ypos,
            text=phase_label,
            showarrow=False,
            font=dict(size=font_size, color="#f1f5f9", family="Montserrat, Arial"),
            xanchor="center", yanchor="middle",
            align="center"
        )

        # EVENTOS modernos: iconos, glow, líneas curvas
        event_emoji = {
            'Target': '🎯',
            'Milestone': '🏁',
            'Barrier': '🚧'
        }
        events = list(p['rows'].iterrows())
        n_ev = len(events)
        if n_ev == 1:
            angles = [0]
        else:
            max_angle = 0.57
            angles = [-max_angle + 2 * max_angle * i / (n_ev - 1) for i in range(n_ev)]
        event_length = 0.95
        base_x = p['start'] + (p['end'] - p['start']) / 2
        base_y = ypos
        for (ev_idx, (irow, row)), angle in zip(enumerate(events), angles):
            direction = p['dir']
            dx = event_length * np.sin(angle)
            dy = direction * event_length * np.cos(angle)
            event_x = base_x + pd.Timedelta(days=dx * 37)
            event_y = base_y - dy
            if direction == 1:
                line_start_y = base_y - BAR_HEIGHT
            else:
                line_start_y = base_y + BAR_HEIGHT

            # Punto evento moderno con icono
            fig.add_trace(go.Scatter(
                x=[event_x], y=[event_y],
                mode='markers+text',
                marker=dict(
                    size=17,
                    color=color_map.get(row['Events'], "#334155"),
                    line=dict(width=3, color="#fff"),
                    opacity=0.98,
                    symbol="circle"
                ),
                text=[event_emoji.get(row['Events'], '')],
                textposition="middle center",
                textfont=dict(size=19),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Swimlane']}</b><br>"
                    f"<b>Event:</b> {row['Events']}<br>"
                    f"<b>Phase:</b> {row['Phase']}<br>"
                    f"<b>Row ID:</b> {row.get('Row ID', '')}<br>"
                    f"<b>Start Date:</b> {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else ''}<br>"
                    + (f"<b>Engagement Action:</b> {row['Engagement Action']}<br>" if row['Events'] == 'Target' and pd.notnull(row.get('Engagement Action', None)) else "")
                    + "<extra></extra>"
                )
            ))

for event, color in color_map.items():
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=16, color=color, line=dict(width=2, color='#fff')),
        legendgroup=event,
        showlegend=True,
        name=event,
        hoverinfo="none"
    ))

if not df_mes.empty and pd.notnull(df_mes['Start Date']).any():
    first_start_date = df_mes['Start Date'].min()
    first_month = first_start_date.replace(day=1)
else:
    first_month = None

if not df_mes.empty and pd.notnull(df_mes['End Date']).any():
    last_end_date = df_mes['End Date'].max()
    last_month = last_end_date.replace(day=1)
else:
    last_month = None
# --- AUTOSCALE X/Y AXES TO FILTERED DATA ---
if not df_mes.empty:
    if first_month == last_month:
        min_x = df_mes['Start Date'].min()
        max_x = df_mes['End Date'].max() + timedelta(days=0.2)
        fig.update_xaxes(range=[min_x, max_x])
        min_y = min(y_pos.values()) - 1.3 - SWIMLANE_SPACING
        max_y = max(y_pos.values()) + 1.3 + SWIMLANE_SPACING
        fig.update_yaxes(range=[min_y, max_y])
    else:
        min_x = df_mes['Start Date'].min()
        max_x = df_mes['End Date'].max() + timedelta(days=6)
        fig.update_xaxes(range=[min_x, max_x])
        min_y = min(y_pos.values()) - 1.3 - SWIMLANE_SPACING
        max_y = max(y_pos.values()) + 1.3 + SWIMLANE_SPACING
        fig.update_yaxes(range=[min_y, max_y])

min_month = df['Month'].min()
max_month = df['Month'].max()
all_months = pd.date_range(min_month, max_month, freq='MS')

if not df_mes.empty and pd.notnull(df_mes['Start Date']).any():
    first_start_date = df_mes['Start Date'].min()
    first_month = first_start_date.replace(day=1)
    first_month_label = first_month.strftime('%b %Y')
else:
    first_month = None
    first_month_label = "Sin datos"

if not df_mes.empty and pd.notnull(df_mes['End Date']).any():
    last_end_date = df_mes['End Date'].max()
    last_month = last_end_date.replace(day=1)
    last_month_label = last_month.strftime('%b %Y')
else:
    last_month = None
    last_month_label = "Sin datos"

if first_month is not None and last_month is not None:
    all_months = pd.date_range(first_month, last_month, freq='MS')
    x_range = [first_month, last_month + pd.offsets.MonthEnd(0)] # incluyente hasta fin de mes
else:
    all_months = pd.date_range(df['Month'].min(), df['Month'].max(), freq='MS')
    x_range = None

fig.update_xaxes(
    showgrid=True,
    gridcolor='#E2E8F0',
    tickmode='array',
    tickvals=all_months,
    ticktext=[d.strftime('%b %Y') for d in all_months],
    title="Date",
    zeroline=False,
    tickfont=dict(size=14, family="Montserrat, Arial", color="#1e293b"),
    title_font=dict(size=16, family="Montserrat, Arial", color="#1e293b"),
    # range=x_range  # Ya lo controla el autoscale arriba
)
fig.update_yaxes(
    tickvals=[y_pos[lane] for lane in swimlanes],
    ticktext=swimlanes,
    showgrid=False,
    zeroline=False,
    showticklabels=True,
    # range=[-1.3 - SWIMLANE_SPACING, SWIMLANE_SPACING * (len(swimlanes)-1) + 1.3 + SWIMLANE_SPACING],  # Ya lo controla el autoscale arriba
    ticks="",
    tickfont=dict(size=14, family="Montserrat, Arial", color="#1e293b")
)
fig.update_layout(
    font_family="Montserrat, Arial",
    font_color="#1e293b",
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
    margin=dict(l=120, r=40, t=60, b=60),
    hoverlabel=dict(bgcolor="white", font_size=15, font_family="Montserrat", font_color="#1e293b"),
    legend=dict(
        title="Event Type",
        orientation="h",
        yanchor="bottom",
        y=1.13,
        xanchor="left",
        x=0.01,
        font=dict(size=14, color="#1e293b"),
        bordercolor="#CBD5E1",
        borderwidth=1
    ),
    height=840,
)

plot_config = {
    "displayModeBar": True,
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "stakeholder_radar",
        "height": 600,
        "width": 800,
        "scale": 3
    },
    "showTips": True,
    "showEditInChartStudio": False,
    "modeBarButtonsToAdd": ["togglelegend"],
    "modeBarButtonsToRemove": ["autoscale"],  # Oculta el botón autoscale
    "responsive": True,
    "showlegendonfullscreen": True
}

with st.container():
    c1, c2 = st.columns([2,1.8])

    with c1:
        st.subheader(f"Swimlane timeline / {first_month_label} - {selected_month_label}")
        st.plotly_chart(fig, use_container_width=True, config=plot_config)

    with c2:
        st.subheader(f"Stakeholder Involvement / {first_month_label} - {selected_month_label}")

        config_col1, config_col2 = st.columns([0.50, 0.50])
        if "show_config_panel" not in st.session_state:
            st.session_state["show_config_panel"] = False
        if "show_config_foco" not in st.session_state:
            st.session_state["show_config_foco"] = False
        with config_col1:
            if st.button("⚙️ Stakeholders", key="open_config_panel", help="Editar pesos de stakeholders"):
                st.session_state["show_config_panel"] = True
        with config_col2:
            if st.button("⚙️ Foco estratégico", key="open_config_foco", help="Editar pesos de foco estratégico"):
                st.session_state["show_config_foco"] = True

        legend_key = "show_legend_radar"
        if legend_key not in st.session_state:
            st.session_state[legend_key] = False

        st.markdown(
            """
            <style>
            div[data-testid="stCheckbox"] label {
                font-size: 10px !important;
                margin-bottom: -8px !important;
                margin-top: -8px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        show_legend = st.checkbox("Mostrar leyenda", value=st.session_state[legend_key], key=legend_key)

        # ------- NUEVO RADAR POR STAKEHOLDER ---------
        colores = ["#f59e42","#eab308","#0284c7","#22d3ee","#22c55e","#e11d48","#7c3aed"]
        radar_traces = []
        for i, stakeholder in enumerate(stakeholder_entities):
            group = df_mes[df_mes['Stakeholder'] == stakeholder] if 'Stakeholder' in df_mes.columns else pd.DataFrame()
            # Para edición: custom_weights por phase+stakeholder
            custom_vals = None
            if "custom_weights" in st.session_state:
                # Si alguna phase está seleccionada, la edición es específica.
                # Pero para el radar, muestra el promedio de todos los pesos para ese stakeholder en el mes.
                # Si hay pesos custom para alguna phase de este stakeholder, úsalos en el promedio.
                # Si no, calcula promedio natural.
                phase_names = df_mes['Phase'].dropna().unique().tolist()
                vals_list = []
                for phase in phase_names:
                    key_id = f"{phase}__{stakeholder}"
                    if key_id in st.session_state['custom_weights']:
                        vals_list.append(st.session_state['custom_weights'][key_id])
                if vals_list:
                    # Promedia los custom si existen
                    avg_vals = [float(np.mean([vals[i] for vals in vals_list])) for i in range(5)]
                    custom_vals = avg_vals
            if group.empty and not custom_vals:
                continue
            if custom_vals:
                vals = custom_vals
            else:
                vals = [0,0,0,0,0]
                for _, row in group.iterrows():
                    st_vals = mapStakeholder(row)
                    vals = [x + y for x, y in zip(vals, st_vals)]
                n = len(group)
                if n: vals = [v/n for v in vals]
            radar_traces.append(go.Scatterpolar(
                r = vals + [vals[0]],
                theta = stakeholder_roles + [stakeholder_roles[0]],
                fill = 'toself',
                name = stakeholder,
                line=dict(color=colores[i%len(colores)], width=3),
                opacity=0.7,
                visible=True
            ))

        radar_layout = go.Layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1.1], color="#1e293b")),
            showlegend=show_legend,
            legend=dict(
                font=dict(family='Montserrat', size=10, color="#1e293b"),
                x=0.99, y=0.01, xanchor='right', yanchor='bottom',
                bordercolor='#e5e7eb', borderwidth=1, bgcolor='#fff',
                orientation='v',
                itemclick='toggle',
                itemdoubleclick='toggleothers'
            ),
            font=dict(color="#1e293b"),
            height=340,
            width=420,
            margin=dict(t=14, l=10, r=10, b=13),
            paper_bgcolor="#fff",
            plot_bgcolor="#fff"
        )

        st.plotly_chart(go.Figure(data=radar_traces, layout=radar_layout), use_container_width=False, config=plot_config)

        # ----------- NUEVO PANEL DE CONFIGURACIÓN DE PESOS ----------
        if st.session_state["show_config_panel"]:
            st.markdown("---")
            st.markdown("#### Configuración de pesos de stakeholders")
            if 'custom_weights' not in st.session_state:
                st.session_state['custom_weights'] = {}

            # 1. Selecciona una fase
            phases = df_mes['Phase'].dropna().unique().tolist()
            selected_phase = st.selectbox("Selecciona una fase", options=[''] + phases, key="config_select_phase")
            if selected_phase:
                # 2. Selecciona un stakeholder
                selected_stakeholder = st.selectbox("Selecciona un stakeholder", options=[''] + stakeholder_entities, key="config_select_stakeholder")
                if selected_stakeholder:
                    group = df_mes[(df_mes['Phase'] == selected_phase) & (df_mes['Stakeholder'] == selected_stakeholder)] if 'Stakeholder' in df_mes.columns else pd.DataFrame()
                    key_id = f"{selected_phase}__{selected_stakeholder}"
                    if key_id in st.session_state['custom_weights']:
                        current_vals = st.session_state['custom_weights'][key_id]
                    else:
                        vals = [0,0,0,0,0]
                        for _, row in group.iterrows():
                            st_vals = mapStakeholder(row)
                            vals = [x + y for x, y in zip(vals, st_vals)]
                        n = len(group)
                        if n: vals = [v/n for v in vals]
                        current_vals = vals
                    with st.form(f"edit_{key_id}"):
                        st.write(f"Ajusta la relevancia de cada rol para el stakeholder **{selected_stakeholder}** en la fase **{selected_phase}**:")
                        new_vals = []
                        for i, role in enumerate(stakeholder_roles):
                            new_vals.append(
                                st.slider(role, min_value=0.0, max_value=1.0, step=0.05, value=float(current_vals[i]), key=f"{key_id}_{role}_config")
                            )
                        col_save, col_close = st.columns([0.5,0.5])
                        with col_save:
                            submit = st.form_submit_button("Guardar cambios")
                        with col_close:
                            close_panel = st.form_submit_button("Cerrar")
                        if submit:
                            st.session_state['custom_weights'][key_id] = new_vals
                            st.success("¡Pesos actualizados para este stakeholder y fase!")
                        if close_panel:
                            st.session_state["show_config_panel"] = False
                else:
                    if st.button("Cerrar", key="cerrar_sin_stakeholder"):
                        st.session_state["show_config_panel"] = False
            else:
                if st.button("Cerrar", key="cerrar_sin_phase"):
                    st.session_state["show_config_panel"] = False

        # ---------- Foco estratégico (igual que antes) ----------
        if st.session_state["show_config_foco"]:
            st.markdown("---")
            st.markdown("#### Configuración de pesos de foco estratégico")
            if 'custom_foco' not in st.session_state:
                df_mes['Macro Fase'] = df_mes['Phase'].map(macroFase)
                fases = ['Desarrollo', 'Implementación', 'Adopción']
                pesos_actuales = df_mes['Macro Fase'].value_counts(normalize=True).reindex(fases, fill_value=0).values.tolist()
                st.session_state['custom_foco'] = pesos_actuales
            with st.form("edit_foco_estrategico"):
                new_foco_vals = []
                fases = ['Desarrollo', 'Implementación', 'Adopción']
                total = sum(st.session_state['custom_foco'])
                st.write("Ajusta el peso relativo de cada foco estratégico (los valores deben sumar 1):")
                for i, fase in enumerate(fases):
                    new_foco_vals.append(
                        st.slider(fase, min_value=0.0, max_value=1.0, step=0.01, value=float(st.session_state['custom_foco'][i]), key=f"foco_{fase}")
                    )
                sum_vals = sum(new_foco_vals)
                if sum_vals != 0:
                    new_foco_vals = [v / sum_vals for v in new_foco_vals]
                col_save, col_close = st.columns([0.5,0.5])
                with col_save:
                    submit_foco = st.form_submit_button("Guardar cambios")
                with col_close:
                    close_foco = st.form_submit_button("Cerrar")
                if submit_foco:
                    st.session_state['custom_foco'] = new_foco_vals
                    st.success("¡Pesos actualizados para foco estratégico!")
                if close_foco:
                    st.session_state["show_config_foco"] = False

        st.subheader(f"Foco estratégico / {first_month_label} - {selected_month_label}")
        df_mes['Macro Fase'] = df_mes['Phase'].map(macroFase)
        fases = ['Desarrollo', 'Implementación', 'Adopción']
        if 'custom_foco' in st.session_state:
            pesos = pd.Series(st.session_state['custom_foco'], index=fases)
        else:
            pesos = df_mes['Macro Fase'].value_counts(normalize=True).reindex(fases, fill_value=0)
        if pesos.sum() > 0:
            st.plotly_chart(go.Figure(
                data=[go.Bar(x=pesos.index, y=pesos.values, marker_color=['#277da1', '#43aa8b', '#f3722c'][:len(pesos)])],
                layout=go.Layout(
                    yaxis=dict(title="Peso relativo", range=[0,1]),
                    xaxis=dict(title=""),
                    height=320,
                    margin=dict(t=30, l=20, r=20, b=40)
                )
            ), use_container_width=True)
        else:
            st.info("Sin datos de foco estratégico para este mes.")

st.markdown("---")
st.caption("Dashboard profesional, powered by Streamlit + Plotly. Cambia el mes a la izquierda para explorar. Puedes editar las aportaciones de los stakeholders seleccionando primero la fase, luego el stakeholder y ajustando los sliders.")
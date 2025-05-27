import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from datetime import datetime

# ----------- CARGA DE DATOS -----------------
st.set_page_config(layout="wide", page_title="Stakeholder Timeline Dashboard")

st.title("Stakeholder Timeline & Strategic Focus")

uploaded = st.sidebar.file_uploader("Carga archivo Csv", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.warning("Por favor, sube tu archivo Cleaned_Timeline_Data.csv para comenzar.")
    st.stop()

# Preprocesar fechas
df['Start Date'] = pd.to_datetime(df['Start Date'])
df['End Date'] = pd.to_datetime(df['End Date'])
df['Point Date'] = df['End Date'].combine_first(df['Start Date']).dt.normalize()
df['Month'] = df['Point Date'].dt.to_period('M').dt.to_timestamp()
months_sorted = sorted(df['Month'].unique())
months_labels = [d.strftime('%b %Y') for d in months_sorted]

# ----------- MAPEOS COHERENTES -----------------
# Stakeholder roles
stakeholder_roles = ['Usuario', 'Prestador', 'Proveedor', 'Facilitador', 'Prescriptor']

def macroFase(phase):
    """Devuelve Desarrollo / Implementación / Adopción siempre."""
    phase = (phase or "").lower()
    # Palabras clave para cada macrofase
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
    # Match por keyword
    if any(k in phase for k in desarrollo_kw):
        return "Desarrollo"
    if any(k in phase for k in implement_kw):
        return "Implementación"
    if any(k in phase for k in adopcion_kw):
        return "Adopción"
    # Heurística: si nada hace match, por defecto “Desarrollo”
    return "Desarrollo"

def mapStakeholder(row):
    v = [0,0,0,0,0]
    if row['Swimlane'] == "User engagement":
        v[0] += 1; v[4] += 1; v[3] += 0.6
    if row['Swimlane'] == "Logistics":
        v[2] += 1; v[3] += 0.8
    if row['Swimlane'] == "Data economy":
        v[1] += 1; v[2] += 0.7
    if pd.notnull(row['Events']) and "barrier" in row['Events'].lower():
        v[3] += 0.6; v[2] += 0.5
    if pd.notnull(row['Events']) and "target" in row['Events'].lower():
        v[0] += 0.7; v[4] += 0.5
    if pd.notnull(row['Events']) and "milestone" in row['Events'].lower():
        v = [x+0.3 for x in v]
    # Normalizar
    maxv = max(v + [1])
    v = [x/maxv for x in v]
    return v

# ----------- INTERFAZ DE USUARIO -------------
with st.sidebar:
    st.subheader("Filtrar por mes/fase")
    mes_idx = st.radio(
        "Elige un mes para ver los eventos y focos estratégicos:",
        options=list(range(len(months_labels))),
        format_func=lambda i: months_labels[i]
    )
    filtro_mes = months_sorted[mes_idx]

# ----------- FILTRO POR MES ------------------
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

swimlanes = ['User engagement', 'Logistics', 'Data economy']
SWIMLANE_SPACING = 1.7
PHASE_SLOT = 0.38

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

# 2. Phases como columnas, con dirección alternante y mayor separación
for lane in swimlanes:
    sub = df_mes[df_mes['Swimlane'] == lane].copy()
    if sub.empty: continue
    # Ordena las phases por start date
    phases = []
    for phase, phase_df in sub.groupby('Phase'):
        start = phase_df['Start Date'].min()
        end = phase_df['End Date'].max()
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
        # Sombra de Phase para efecto visual elegante
        fig.add_trace(go.Scatter(
            x=[p['start'], p['end']],
            y=[ypos-0.06, ypos-0.06],
            mode='lines',
            line=dict(color='#94a3b8', width=10, dash='dot', shape='spline'),
            showlegend=False,
            hoverinfo='skip',
            opacity=0.13
        ))
            # Texto Phase
        # Barra phase más gruesa
        BAR_HEIGHT = 0.23  # Hazlo más grueso para meter el texto
        fig.add_trace(go.Scatter(
            x=[p['start'], p['end'], p['end'], p['start'], p['start']],
            y=[ypos-BAR_HEIGHT, ypos-BAR_HEIGHT, ypos+BAR_HEIGHT, ypos+BAR_HEIGHT, ypos-BAR_HEIGHT],
            fill="toself",
            fillcolor="#0f172a",
            line=dict(width=0),
            mode='lines',
            showlegend=False,
            hoverinfo='skip'
        ))

            # Calcula la anchura de la barra en segundos (o días según tu preferencia)
        bar_width = (p['end'] - p['start']).total_seconds()
        text = p['name']
        char_count = len(text)
        x_unit_to_px = 30  # ajusta según tu escala
        font_size_max = 12
        text_width_est = char_count * font_size_max * 0.6
        bar_width_px = bar_width * x_unit_to_px

        if text_width_est > bar_width_px:
            font_size = max(8, int(bar_width_px / (char_count * 0.6)))
        else:
            font_size = font_size_max

        # Texto phase centrado en la barra, sin fondo
               
        xtext = p['start'] + (p['end'] - p['start']) / 2
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,-1), (1,1), (-1,1), (1,-1)]:
            fig.add_annotation(
                x=xtext,
                y=ypos + dy * 0.01,  # ajusta 0.01 según tu escala de y
                text=f"<b>{p['name']}</b>",
                showarrow=False,
                font=dict(size=font_size, color="#000", family="Montserrat, Arial"),
                xanchor="center", yanchor="middle",
                align="center",
                opacity=1,
                # borderpad=0, # puedes ajustar si quieres
            )
        # Y encima el texto real en blanco/grisáceo
        fig.add_annotation(
            x=xtext,
            y=ypos,
            text=f"<b>{p['name']}</b>",
            showarrow=False,
            font=dict(size=font_size, color="#f1f5f9", family="Montserrat, Arial"),
            xanchor="center", yanchor="middle",
            align="center"
        )



                # Eventos en abanico
        events = list(p['rows'].iterrows())
        n_ev = len(events)
        if n_ev == 1:
            angles = [0]
        else:
            max_angle = 0.48
            angles = [-max_angle + 2 * max_angle * i / (n_ev - 1) for i in range(n_ev)]
        event_length = 0.43

        # Centro de la barra para el abanico
        base_x = p['start'] + (p['end'] - p['start']) / 2
        base_y = ypos

        for (ev_idx, (irow, row)), angle in zip(enumerate(events), angles):
            direction = p['dir']
            dx = event_length * np.sin(angle)
            dy = direction * event_length * np.cos(angle)
            event_x = base_x + pd.Timedelta(days=dx * 37)
            event_y = base_y - dy

            # Elegir borde de la barra según direction
            if direction == 1:
                line_start_y = base_y - BAR_HEIGHT  # ABAJO
            else:
                line_start_y = base_y + BAR_HEIGHT  # ARRIBA

            fig.add_trace(go.Scatter(
                x=[base_x, event_x], y=[line_start_y, event_y],
                mode="lines",
                line=dict(
                    color=rgba_map.get(row['Events'], "rgba(100,116,139,0.48)"),
                    width=3.5,
                    shape="spline"
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Sombra del marcador
            fig.add_trace(go.Scatter(
                x=[event_x], y=[event_y],
                mode='markers',
                marker=dict(size=28, color="#CBD5E1", opacity=0.30, line=dict(width=0)),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Punto evento principal
            fig.add_trace(go.Scatter(
                x=[event_x], y=[event_y],
                mode='markers',
                marker=dict(size=18, color=color_map.get(row['Events'], "#334155"), line=dict(width=3, color="#fff"), opacity=0.96),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Swimlane']}</b><br>"
                    f"<b>Event:</b> {row['Events']}<br>"
                    f"<b>Phase:</b> {row['Phase']}<br>"
                    f"<b>Row ID:</b> {row.get('Row ID', '')}<br>"
                    + (f"<b>Engagement Action:</b> {row['Engagement Action']}<br>" if row['Events'] == 'Target' and pd.notnull(row.get('Engagement Action', None)) else "")
                    + "<extra></extra>"
                )
            ))


# Leyenda de eventos
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

# Ejes y layout
min_month = df['Month'].min()
max_month = df['Month'].max()
all_months = pd.date_range(min_month, max_month, freq='MS')

fig.update_xaxes(
    showgrid=True,
    gridcolor='#E2E8F0',
    tickmode='array',
    tickvals=all_months,
    ticktext=[d.strftime('%b %Y') for d in all_months],
    title="Date (End Date)",
    zeroline=False,
    tickfont=dict(size=14, family="Montserrat, Arial", color="#1e293b"),
    title_font=dict(size=16, family="Montserrat, Arial", color="#1e293b")
)
fig.update_yaxes(
    tickvals=[y_pos[lane] for lane in swimlanes],
    ticktext=swimlanes,
    showgrid=False,
    zeroline=False,
    showticklabels=True,
    range=[-0.8 - SWIMLANE_SPACING, SWIMLANE_SPACING * (len(swimlanes)-1) + 0.8 + SWIMLANE_SPACING],
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
    height=720,
)


# ---------- DASHBOARD LATERAL -----------------
with st.container():
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader(f"Swimlane timeline — {months_labels[mes_idx]}")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader(f"Stakeholder Involvement — {months_labels[mes_idx]}")
        # Radar chart# ----------- TIMELINE (Swimlane) --------------
        groups = df_mes.groupby('Row ID')
        radar_traces = []
        colores = ["#f59e42","#eab308","#0284c7","#22d3ee","#22c55e","#e11d48","#7c3aed"]
        for i, (gid, g) in enumerate(groups):
            vals = [0,0,0,0,0]
            for _, row in g.iterrows():
                st_vals = mapStakeholder(row)
                vals = [x + y for x,y in zip(vals, st_vals)]
            n = len(g)
            if n: vals = [v/n for v in vals]
            radar_traces.append(go.Scatterpolar(
                r = vals + [vals[0]],
                theta = stakeholder_roles + [stakeholder_roles[0]],
                fill = 'toself',
                name = gid[:35] + ("..." if len(gid)>35 else ""),
                line=dict(color=colores[i%len(colores)], width=3),
                opacity=0.6
            ))
        radar_layout = go.Layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,1], color="#1e293b")),
    showlegend=True,
    legend=dict(
        font=dict(family='Montserrat', size=9, color="#1e293b"),  # Pequeño y oscuro
        x=1.02, y=1, xanchor='left', yanchor='top',
        bordercolor='#e5e7eb', borderwidth=1, bgcolor='#fff'
    ),
    font=dict(color="#1e293b"),
    height=320,
    margin=dict(t=30, l=20, r=20, b=20),
    paper_bgcolor="#fff",
    plot_bgcolor="#fff"
)

        if radar_traces:
            st.plotly_chart(go.Figure(data=radar_traces, layout=radar_layout), use_container_width=True)
        else:
            st.info("Sin datos de stakeholders para este mes.")

        # Bar chart foco estratégico
        st.subheader(f"Foco estratégico — {months_labels[mes_idx]}")
        df_mes['Macro Fase'] = df_mes['Phase'].map(macroFase)
        pesos = df_mes['Macro Fase'].value_counts(normalize=True).sort_index()
        if len(pesos):
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
st.caption("Dashboard profesional, powered by Streamlit + Plotly. Cambia el mes a la izquierda para explorar.")

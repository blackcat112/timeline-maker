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
    if pd.notnull(row['Events']) and "barrier" in row['Events'].lower():
        v[3] += 0.6; v[2] += 0.5
    if pd.notnull(row['Events']) and "target" in row['Events'].lower():
        v[0] += 0.7; v[4] += 0.5
    if pd.notnull(row['Events']) and "milestone" in row['Events'].lower():
        v = [x+0.3 for x in v]
    maxv = max(v + [1])
    v = [x/maxv for x in v]
    if custom_weights is not None:
        v = custom_weights
    return v

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
swimlanes = ['User engagement', 'Logistics', 'Data economy']

# MÁS ESPACIO entre swimlanes y phases
SWIMLANE_SPACING = 2.4  # antes 1.7
PHASE_SLOT = 0.65       # antes 0.38

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
    phases = []
    for phase, phase_df in sub.groupby('Phase'):
        # Aseguramos que los valores NaT no se cuenten para el rango
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
        # Barra phase más gruesa y visible: azul claro con borde blanco
        BAR_HEIGHT = 0.33  # Más grueso
        fig.add_trace(go.Scatter(
            x=[p['start'], p['end'], p['end'], p['start'], p['start']],
            y=[ypos-BAR_HEIGHT, ypos-BAR_HEIGHT, ypos+BAR_HEIGHT, ypos+BAR_HEIGHT, ypos-BAR_HEIGHT],
            fill="toself",
            fillcolor="#60a5fa",  # azul claro
            line=dict(width=3, color="#fff"),  # borde blanco
            mode='lines',
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

        # --- EVENTOS modernos: iconos, glow, líneas curvas ---
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
        event_length = 0.56
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

            # Línea curva con efecto moderno
            fig.add_trace(go.Scatter(
                x=[base_x, event_x], y=[line_start_y, event_y],
                mode="lines",
                line=dict(
                    color=rgba_map.get(row['Events'], "rgba(100,116,139,0.28)"),
                    width=5,
                    shape="spline"
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Glow/Sombra
            fig.add_trace(go.Scatter(
                x=[event_x], y=[event_y],
                mode='markers',
                marker=dict(
                    size=38, 
                    color="#000", 
                    opacity=0.17, 
                    line=dict(width=0)
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Punto evento moderno con icono
            fig.add_trace(go.Scatter(
                x=[event_x], y=[event_y],
                mode='markers+text',
                marker=dict(
                    size=22,
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
    range=[-1.3 - SWIMLANE_SPACING, SWIMLANE_SPACING * (len(swimlanes)-1) + 1.3 + SWIMLANE_SPACING],
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

with st.container():
    c1, c2 = st.columns([2,1.8])

    with c1:
        st.subheader(f"Swimlane timeline — {selected_month_label}")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader(f"Stakeholder Involvement — {selected_month_label}")

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

        radar_traces = []
        colores = ["#f59e42","#eab308","#0284c7","#22d3ee","#22c55e","#e11d48","#7c3aed"]
        grupos = df_mes.groupby('Row ID')
        for i, (gid, g) in enumerate(grupos):
            if gid in st.session_state.get('custom_weights', {}):
                vals = st.session_state['custom_weights'][gid]
            else:
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
                opacity=0.7,
                visible=True
            ))

        radar_layout = go.Layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1.1], color="#1e293b")),
            showlegend=show_legend,
            legend=dict(
                font=dict(family='Montserrat', size=8, color="#1e293b"),
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
            "modeBarButtonsToRemove": [],
            "responsive": True,
            "showlegendonfullscreen": True
        }

        st.plotly_chart(go.Figure(data=radar_traces, layout=radar_layout), use_container_width=False, config=plot_config)

        if st.session_state["show_config_panel"]:
            st.markdown("---")
            st.markdown("#### Configuración de pesos de stakeholders")
            if 'custom_weights' not in st.session_state:
                st.session_state['custom_weights'] = {}
            grupos = df_mes.groupby('Row ID')
            eventos_lista = []
            for gid, g in grupos:
                nombre = gid[:35] + ("..." if len(gid)>35 else "")
                eventos_lista.append(nombre)
            selected_g = st.selectbox("Selecciona un evento para editar stakeholders", options=[''] + eventos_lista, key="config_select_event")
            selected_gid = None
            if selected_g != '':
                for gid in grupos.groups.keys():
                    if selected_g == gid[:35] + ("..." if len(gid)>35 else ""):
                        selected_gid = gid
                        break
            if selected_gid is not None:
                g = grupos.get_group(selected_gid)
                if selected_gid in st.session_state['custom_weights']:
                    current_vals = st.session_state['custom_weights'][selected_gid]
                else:
                    vals = [0,0,0,0,0]
                    for _, row in g.iterrows():
                        st_vals = mapStakeholder(row)
                        vals = [x + y for x,y in zip(vals, st_vals)]
                    n = len(g)
                    if n: vals = [v/n for v in vals]
                    current_vals = vals
                with st.form(f"edit_{selected_gid}"):
                    st.write("Ajusta la relevancia de cada stakeholder para este evento:")
                    new_vals = []
                    for i, role in enumerate(stakeholder_roles):
                        new_vals.append(st.slider(role, min_value=0.0, max_value=1.0, step=0.05, value=float(current_vals[i]), key=f"{selected_gid}_{role}_config"))
                    col_save, col_close = st.columns([0.5,0.5])
                    with col_save:
                        submit = st.form_submit_button("Guardar cambios")
                    with col_close:
                        close_panel = st.form_submit_button("Cerrar")
                    if submit:
                        st.session_state['custom_weights'][selected_gid] = new_vals
                        st.success("¡Pesos actualizados para este evento!")
                    if close_panel:
                        st.session_state["show_config_panel"] = False
            else:
                if st.button("Cerrar", key="cerrar_sin_seleccion"):
                    st.session_state["show_config_panel"] = False

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

        st.subheader(f"Foco estratégico — {selected_month_label}")
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
st.caption("Dashboard profesional, powered by Streamlit + Plotly. Cambia el mes a la izquierda para explorar. Puedes editar las aportaciones de los stakeholders seleccionando eventos en la leyenda de radar o usando los paneles de configuración.")
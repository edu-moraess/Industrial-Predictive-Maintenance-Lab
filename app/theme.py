"""
Plotly Industrial Theme Configuration
"""

def apply_industrial_plotly_theme(fig, height=280):
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="JetBrains Mono", color="#94A3B8", size=10),
        margin=dict(l=15, r=15, t=30, b=15),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            gridcolor="#253044",
            linecolor="#253044",
            zerolinecolor="#253044",
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            gridcolor="#253044",
            linecolor="#253044",
            zerolinecolor="#253044",
            tickfont=dict(size=9)
        )
    )
    return fig

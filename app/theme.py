"""
Plotly Industrial Theme Configuration
"""

def apply_industrial_plotly_theme(fig, height=260):
    fig.update_layout(
        paper_bgcolor="#171A21",
        plot_bgcolor="#171A21",
        font=dict(family="Arial, sans-serif", color="#9A9FA8", size=10),
        margin=dict(l=10, r=10, t=25, b=10),
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
            gridcolor="#2A2F38",
            linecolor="#2A2F38",
            zerolinecolor="#2A2F38",
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            gridcolor="#2A2F38",
            linecolor="#2A2F38",
            zerolinecolor="#2A2F38",
            tickfont=dict(size=9)
        )
    )
    return fig

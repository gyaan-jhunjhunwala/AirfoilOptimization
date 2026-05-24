import streamlit as st
import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad
import matplotlib.pyplot as plt
import neuralfoil as nf


# 1. PAGE SETUP & STYLING
# ----------------------------------------

st.set_page_config(
    page_title="Airfoil Optimizer & Visualizer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@300;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        color: #F0F2F6;
        font-weight: 600;
        border-bottom: 2px solid #00C6FF;
        padding-bottom: 5px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(0, 198, 255, 0.4);
        box-shadow: 0 12px 40px 0 rgba(0, 198, 255, 0.15);
        transform: translateY(-2px);
    }
    
    .metric-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #A1A8B8;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    .metric-value-opt {
        color: #00C6FF;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3) !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(0, 198, 255, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


# 2. AIRFOIL MATH & GEOMETRY
# ----------------------------------------

def generate_naca_coords(camber, camber_pos, thickness, resolution=100):
    beta = np.linspace(0, np.pi, resolution)
    x_chord = 0.5 * (1.0 - np.cos(beta))
    
    mean_camber = np.zeros_like(x_chord)
    camber_slope = np.zeros_like(x_chord)
    
    if camber_pos > 0 and camber > 0:
        front_part = x_chord <= camber_pos
        back_part = x_chord > camber_pos
        
        mean_camber[front_part] = (camber / (camber_pos**2)) * (
            2.0 * camber_pos * x_chord[front_part] - x_chord[front_part]**2
        )
        camber_slope[front_part] = (2.0 * camber / (camber_pos**2)) * (
            camber_pos - x_chord[front_part]
        )
        
        mean_camber[back_part] = (camber / ((1.0 - camber_pos)**2)) * (
            (1.0 - 2.0 * camber_pos) + 2.0 * camber_pos * x_chord[back_part] - x_chord[back_part]**2
        )
        camber_slope[back_part] = (2.0 * camber / ((1.0 - camber_pos)**2)) * (
            camber_pos - x_chord[back_part]
        )
        
    thickness_dist = 5.0 * thickness * (
        0.2969 * np.sqrt(x_chord) - 
        0.1260 * x_chord - 
        0.3516 * (x_chord**2) + 
        0.2843 * (x_chord**3) - 
        0.1015 * (x_chord**4)
    )
    
    theta = np.arctan(camber_slope)
    
    x_upper = x_chord - thickness_dist * np.sin(theta)
    y_upper = mean_camber + thickness_dist * np.cos(theta)
    
    x_lower = x_chord + thickness_dist * np.sin(theta)
    y_lower = mean_camber - thickness_dist * np.cos(theta)
    
    upper_points = np.stack([x_upper[::-1], y_upper[::-1]], axis=1)
    lower_points = np.stack([x_lower[1:], y_lower[1:]], axis=1) 
    
    return np.concatenate([upper_points, lower_points], axis=0)


# 3. AERODYNAMICS BACKEND
# ----------------------------------------

def compute_ideal_lift(camber, camber_pos, alpha_rad):
    if camber == 0 or camber_pos == 0:
        alpha_zero_lift = 0.0
    else:
        def glauert_integrand(theta):
            x = 0.5 * (1.0 - np.cos(theta))
            slope = (
                (2.0 * camber / (camber_pos**2)) * (camber_pos - x)
                if x <= camber_pos
                else (2.0 * camber / ((1.0 - camber_pos)**2)) * (camber_pos - x)
            )
            return slope * (1.0 - np.cos(theta))
        
        integral_result, _ = quad(glauert_integrand, 0.0, np.pi)
        alpha_zero_lift = integral_result / np.pi
        
    lift_coef = 2.0 * np.pi * (alpha_rad - alpha_zero_lift)
    return lift_coef, alpha_zero_lift

def estimate_friction_drag(camber, camber_pos, thickness, cl_ideal):
    cd_parasitic = 0.0050 + 0.01 * thickness + 0.10 * (thickness**2) + 0.00008 / thickness
    k_induced = 0.005 + 0.05 * thickness
    cl_min_drag = 1.5 * camber
    cd_lift_penalty = k_induced * ((cl_ideal - cl_min_drag)**2)
    return cd_parasitic + cd_lift_penalty

def apply_flow_separation_penalties(thickness, effective_alpha_deg):
    drag_penalty = 0.0
    lift_scale = 1.0
    is_stalled = False
    
    if effective_alpha_deg > 12.0:
        lift_scale = 0.5
        drag_penalty += 0.08
        is_stalled = True
        
    if thickness < 0.10:
        clamped_t = max(thickness, 0.005)
        drag_penalty += 0.008 * ((0.10 - clamped_t) / 0.10) + 0.02 * (
            (0.10 - clamped_t) / clamped_t
        )**2
        
    return lift_scale, drag_penalty, is_stalled


# 4. DATA WRAPPING & ENGINE ROUTING
# ----------------------------------------

def analyze_airfoil(camber, camber_pos, thickness, alpha_deg, aspect_ratio=12.0, reynolds=1e6):
    alpha_rad = np.radians(alpha_deg)
    cl_ideal, alpha_l0_rad = compute_ideal_lift(camber, camber_pos, alpha_rad)
    alpha_l0_deg = np.degrees(alpha_l0_rad)
    cd_empirical = estimate_friction_drag(camber, camber_pos, thickness, cl_ideal)
    
    coordinates = generate_naca_coords(camber, camber_pos, thickness)
    cl_final = cl_ideal
    cd_final = cd_empirical
    neuralfoil_worked = False
    
    try:
        nf_data = nf.get_aero_from_coordinates(
            coordinates, alpha=alpha_deg, Re=reynolds, model_size='xlarge'
        )
        cl_predicted = float(nf_data['CL'][0])
        cd_predicted = float(nf_data['CD'][0])
        
        if cd_predicted > 0:
            cl_final = cl_predicted
            cd_final = cd_predicted
            neuralfoil_worked = True
    except Exception:
        pass
        
    if neuralfoil_worked:
        min_drag_floor = 0.0050 + 0.01 * thickness + 0.10 * thickness**2 + 0.00008 / thickness
        if cd_final < min_drag_floor:
            cd_final = min_drag_floor
            
    effective_alpha = alpha_deg - alpha_l0_deg
    lift_scale, drag_penalty, is_stalled = apply_flow_separation_penalties(thickness, effective_alpha)
    
    cl_ideal *= lift_scale
    cl_final *= lift_scale
    cd_empirical += drag_penalty
    cd_final += drag_penalty
    
    oswald_eff = 0.85
    induced_drag_factor = 1.0 / (np.pi * aspect_ratio * oswald_eff)
    
    cd_ideal_total = cd_empirical + (cl_ideal**2) * induced_drag_factor
    cd_final_total = cd_final + (cl_final**2) * induced_drag_factor
    
    eff_ideal = cl_ideal / cd_ideal_total if cd_ideal_total > 0 else 0.0
    eff_final = cl_final / cd_final_total if cd_final_total > 0 else 0.0
        
    return {
        'cl_theory': cl_ideal,
        'cd_theory': cd_ideal_total,
        'eff_theory': eff_ideal,
        'alpha_l0_deg': alpha_l0_deg,
        'alpha_eff_deg': effective_alpha,
        'stalled': is_stalled,
        'cl_nf': cl_final,
        'cd_nf': cd_final_total,
        'eff_nf': eff_final,
        'nf_success': neuralfoil_worked
    }


# 5. MATH OPTIMIZER LOGIC
# ----------------------------------------

def optimize_airfoil_efficiency(init_camber, init_pos, init_thick, init_alpha, aspect_ratio=12.0, reynolds=1e6, min_cl_target=0.4):
    def loss_function(params):
        c, p, t, a = params
        metrics = analyze_airfoil(c, p, t, a, aspect_ratio, reynolds)
        if metrics['cl_nf'] <= 0.01 or metrics['cd_nf'] <= 0:
            return 1e6
        return -metrics['eff_nf']
        
    def lift_constraint(params):
        c, p, t, a = params
        metrics = analyze_airfoil(c, p, t, a, aspect_ratio, reynolds)
        return metrics['cl_nf'] - min_cl_target

    search_bounds = [
        (0.00, 0.06),  
        (0.25, 0.60),  
        (0.10, 0.18),  
        (1.0,  8.0)    
    ]
    
    clamped_camber = np.clip(init_camber, search_bounds[0][0], search_bounds[0][1])
    clamped_pos = np.clip(init_pos, search_bounds[1][0], search_bounds[1][1])
    clamped_thick = np.clip(init_thick, search_bounds[2][0], search_bounds[2][1])
    clamped_alpha = np.clip(init_alpha, search_bounds[3][0], search_bounds[3][1])
    
    initial_guess = [clamped_camber, clamped_pos, clamped_thick, clamped_alpha]
    constraints = [{'type': 'ineq', 'fun': lift_constraint}]
    
    raw_result = opt.minimize(
        loss_function, 
        initial_guess, 
        method='SLSQP', 
        bounds=search_bounds, 
        constraints=constraints,
        options={'ftol': 1e-4, 'maxiter': 60}
    )
    
    opt_c, opt_p, opt_t, opt_a = raw_result.x
    opt_metrics = analyze_airfoil(opt_c, opt_p, opt_t, opt_a, aspect_ratio, reynolds)
    
    return {
        'opt_m': opt_c,
        'opt_p': opt_p,
        'opt_t': opt_t,
        'opt_alpha': opt_a,
        'opt_cl': opt_metrics['cl_nf'],
        'opt_cd': opt_metrics['cd_nf'],
        'opt_efficiency': opt_metrics['eff_nf'],
        'success': raw_result.success
    }


# 6. STREAMLIT FRONTEND USER INTERFACE
# ----------------------------------------

if "warnings_acknowledged" not in st.session_state:
    st.session_state.warnings_acknowledged = False

@st.dialog("⚠️ Critical Solver Constraints & Limitations")
def show_solver_warnings_modal():
    st.markdown("""
    ### 
    1. **Low Reynolds Number Constraints ($Re \le 10^5$)** : 
       The integrated machine learning model (`NeuralFoil`) is tuned for transitional and turbulent flows. Operating near or below $Re = 10^5$ may introduce numerical instability or default to the empirical backend.
    
    2. **Conflicting Lift Targets** : 
       If you set a high **Minimum Target Lift ($C_l$)** while restricting the shape controls to a low maximum camber and thickness, the mathematical optimizer (`SLSQP`) may fail to find a valid solution space, resulting in an efficiency drop.
    
    3. **The Artificial Stall Boundary Threshold ($\\alpha_{\\text{eff}} > 12^\\circ$)** : 
       To prevent non-physical performance predictions, a sharp flow-separation penalty cuts lift by 50% and scales up drag dramatically if the effective angle of attack exceeds $12^\\circ$. The optimizer will naturally steer away from these zones.
    """)
    
    if st.button("I understand these limitations"):
        st.session_state.warnings_acknowledged = True
        st.rerun()

if not st.session_state.warnings_acknowledged:
    show_solver_warnings_modal()
    st.stop()

st.markdown('<div class="main-title">Airfoil Dashboard</div>', unsafe_allow_html=True)
st.markdown("##### Design, test, and automatically optimize customized 4-digit NACA profiles live", unsafe_allow_html=True)
st.write("---")

if "opt_results" not in st.session_state:
    st.session_state.opt_results = None
if "previous_inputs" not in st.session_state:
    st.session_state.previous_inputs = {}

with st.sidebar:
    st.markdown("### 🛠️ Airfoil Shape Parameters")
    
    aoa_deg = st.slider(
        "Angle of Attack (Degrees)", 
        min_value=1.0, max_value=8.0, value=4.0, step=0.5,
        help="The angle between the oncoming air flow vector and the chord line axis."
    )
    
    camber_pct = st.slider(
        "Maximum Camber (%)", 
        min_value=0.0, max_value=6.0, value=2.0, step=0.5,
        help="The maximum distance between the mean camber line and the chord line, expressed as a percentage of the chord length."
    )
    camber_fraction = camber_pct / 100.0
    
    camber_pos_ref = st.radio(
        "Reference Camber Position From:",
        ["Leading Edge (Standard)", "Trailing Edge (Rearwards)"],
        horizontal=True
    )
    
    if camber_pos_ref == "Leading Edge (Standard)":
        camber_pos_pct = st.slider(
            "Camber Position (%)", 
            min_value=25.0, max_value=60.0, value=40.0, step=5.0,
            help="The distance from the leading edge to the location of maximum camber, expressed as a percentage of the chord length."
        )
        camber_pos_fraction = camber_pos_pct / 100.0
    else:
        camber_pos_from_end = st.slider(
            "Camber Position from Rear (%)", 
            min_value=40.0, max_value=75.0, value=60.0, step=5.0,
            help="Alternative structural reference measuring maximum camber location backward from the trailing edge."
        )
        camber_pos_fraction = 1.0 - (camber_pos_from_end / 100.0)
        
    thickness_pct = st.slider(
        "Maximum Thickness (%)", 
        min_value=10.0, max_value=18.0, value=12.0, step=0.5, 
        help="The maximum height of the profile normal to the chord line, expressed as a percentage of the chord length."        
    )
    thickness_fraction = thickness_pct / 100.0
    
    st.markdown("### 🛬 Flying Conditions")
    
    reynolds_selection = st.slider(
        "Reynolds Number (Re)",
        min_value=1e5,
        max_value=3e6,
        value=1e6,
        step=1e5,
        format="%e",
        help="A dimensionless parameter representing the ratio of inertial forces to viscous forces."
    )
    
    wing_ar = st.slider(
        "Wing Aspect Ratio (AR)", 
        min_value=5.0, max_value=25.0, value=12.0, step=0.5,
        help="The ratio of wing span to its mean chord length."
    )
    lift_constraint_target = st.slider(
        "Minimum Target Lift Coefficient (Cl)", 
        min_value=0.1, max_value=0.8, value=0.4, step=0.05,
        help="The solver must find a geometric shape that satisfies or exceeds this lift coefficient while simultaneously minimizing total drag."
    )
    
    trigger_optimization = st.button("Optimize this airfoil design")

current_inputs = {
    "aoa": aoa_deg, "cam": camber_fraction, "pos": camber_pos_fraction, 
    "thick": thickness_fraction, "re": reynolds_selection, "ar": wing_ar, "target": lift_constraint_target
}

if st.session_state.previous_inputs and st.session_state.previous_inputs != current_inputs:
    st.session_state.opt_results = None  

st.session_state.previous_inputs = current_inputs

active_analysis = analyze_airfoil(camber_fraction, camber_pos_fraction, thickness_fraction, aoa_deg, wing_ar, reynolds_selection)

if trigger_optimization:
    with st.spinner("Calculating optimal aerodynamic surfaces..."):
        st.session_state.opt_results = optimize_airfoil_efficiency(
            camber_fraction, camber_pos_fraction, thickness_fraction, aoa_deg, wing_ar, reynolds_selection, min_cl_target=lift_constraint_target
        )

if st.session_state.opt_results:
    display_col1, display_col2 = st.columns([1, 1])
else:
    display_col1, display_col2 = st.columns([2, 1])

# Current Design Performance Stats
with display_col1:
    st.markdown('<div class="section-title">Current Airfoil Metrics</div>', unsafe_allow_html=True)
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">Lift Coefficient (C<sub>l</sub>)</div>
            <div class="metric-value">{active_analysis['cl_nf']:.3f}</div>
            <div style="font-size:0.85rem; color:#A1A8B8; margin-top:5px;">
                Inviscid Theory: {active_analysis['cl_theory']:.3f}<br>
                Zero-Lift AoA: {active_analysis['alpha_l0_deg']:.1f}°
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">Drag Coefficient (C<sub>d</sub>)</div>
            <div class="metric-value">{active_analysis['cd_nf']:.5f}</div>
            <div style="font-size:0.85rem; color:#A1A8B8; margin-top:5px;">
                Empirical Estimate: {active_analysis['cd_theory']:.5f}<br>
                Flow State (Re): {reynolds_selection:.0e}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">Aerodynamic Efficiency (C<sub>l</sub>/C<sub>d</sub>)</div>
            <div class="metric-value" style="color: #00FFB2;">{active_analysis['eff_nf']:.2f}</div>
            <div style="font-size:0.85rem; color:#A1A8B8; margin-top:5px;">
                Ideal Theory C<sub>L</sub>/C<sub>D</sub>: {active_analysis['eff_theory']:.2f}<br>
                Current Angle: {aoa_deg:.2f}°
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        naca_name_string = f"NACA {int(camber_fraction*100)}{int(camber_pos_fraction*10)}{int(thickness_fraction*100):02d}"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">NACA Profile Code</div>
            <div class="metric-value">{naca_name_string}</div>
            <div style="font-size:0.85rem; color:#A1A8B8; margin-top:5px;">
                Camber: {camber_fraction*100:.1f}% @ {camber_pos_fraction*100:.0f}% chord<br>
                Thickness: {thickness_fraction*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

opt_naca_code = ""

with display_col2:
    st.markdown('<div class="section-title">Optimized Airfoil Metrics</div>', unsafe_allow_html=True)
    if st.session_state.opt_results:
        optimized_dataset = st.session_state.opt_results
        opt_col1, opt_col2 = st.columns(2)
        
        opt_naca_code = f"NACA {int(round(optimized_dataset['opt_m']*100))}{int(round(optimized_dataset['opt_p']*10))}{int(round(optimized_dataset['opt_t']*100)):02d}"
        efficiency_gain = ((optimized_dataset['opt_efficiency'] - active_analysis['eff_nf']) / active_analysis['eff_nf']) * 100
        
        with opt_col1:
            st.markdown(f"""
            <div class="glass-card" style="border-color: rgba(0, 198, 255, 0.5);">
                <div class="metric-title">Optimal Profile Code</div>
                <div class="metric-value metric-value-opt">{opt_naca_code}</div>
                <div style="font-size:0.85rem; color:#A1A8B8; margin-top:5px;">
                    Camber: {optimized_dataset['opt_m']*100:.2f}% @ {optimized_dataset['opt_p']*100:.1f}% chord<br>
                    Thickness: {optimized_dataset['opt_t']*100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="glass-card" style="border-color: rgba(0, 198, 255, 0.5);">
                <div class="metric-title">Optimal Angle of Attack</div>
                <div class="metric-value metric-value-opt">{optimized_dataset['opt_alpha']:.2f}°</div>
                <div style="font-size:0.85rem; color:#A1A8B8; margin-top:5px;">
                    Angle in Radians: {np.radians(optimized_dataset['opt_alpha']):.4f} rad
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with opt_col2:
            st.markdown(f"""
            <div class="glass-card" style="border-color: #00FFB2;">
                <div class="metric-title">Maximized Efficiency (C<sub>l</sub>/C<sub>d</sub>)</div>
                <div class="metric-value" style="color: #00FFB2;">{optimized_dataset['opt_efficiency']:.2f}</div>
                <div style="font-size:0.85rem; color:#00FFB2; font-weight: 600; margin-top:5px;">
                    Net Gain: +{efficiency_gain:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-title">Optimized Coefficients</div>
                <div class="metric-value" style="font-size:1.8rem;">C<sub>L</sub>: {optimized_dataset['opt_cl']:.3f}</div>
                <div class="metric-value" style="font-size:1.8rem;">C<sub>D</sub>: {optimized_dataset['opt_cd']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("💡 Adjust your values on the sidebar and hit 'Optimize this airfoil design'.")


# 7. VISUALIZATIONS & PERFORMANCE PLOTS
# ----------------------------------------

st.markdown('<div class="section-title">📊 Airfoil Geometry Comparison Plot</div>', unsafe_allow_html=True)

base_coords = generate_naca_coords(camber_fraction, camber_pos_fraction, thickness_fraction)
chord_line_axis = np.linspace(0, 1, 100)
base_camber_curve = np.zeros_like(chord_line_axis)

if camber_pos_fraction > 0 and camber_fraction > 0:
    front_curve = chord_line_axis <= camber_pos_fraction
    back_curve = chord_line_axis > camber_pos_fraction
    base_camber_curve[front_curve] = (camber_fraction / (camber_pos_fraction**2)) * (2.0 * camber_pos_fraction * chord_line_axis[front_curve] - chord_line_axis[front_curve]**2)
    base_camber_curve[back_curve] = (camber_fraction / ((1.0 - camber_pos_fraction)**2)) * ((1.0 - 2.0 * camber_pos_fraction) + 2.0 * camber_pos_fraction * chord_line_axis[back_curve] - chord_line_axis[back_curve]**2)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#1e2430')

ax.plot(chord_line_axis, base_camber_curve, color='#A1A8B8', linestyle='--', label='Original Camber Spine', alpha=0.7)
ax.plot(base_coords[:, 0], base_coords[:, 1], color='#FF4B4B', linewidth=2.5, label=f'Original Shape ({naca_name_string})')
ax.fill(base_coords[:, 0], base_coords[:, 1], color='#FF4B4B', alpha=0.15)

if st.session_state.opt_results:
    opt_results_data = st.session_state.opt_results
    optimized_coords = generate_naca_coords(opt_results_data['opt_m'], opt_results_data['opt_p'], opt_results_data['opt_t'])
    opt_camber_curve = np.zeros_like(chord_line_axis)
    opt_p_val, opt_m_val = opt_results_data['opt_p'], opt_results_data['opt_m']
    
    if opt_p_val > 0 and opt_m_val > 0:
        front_curve = chord_line_axis <= opt_p_val
        back_curve = chord_line_axis > opt_p_val
        opt_camber_curve[front_curve] = (opt_m_val / (opt_p_val**2)) * (2.0 * opt_p_val * chord_line_axis[front_curve] - chord_line_axis[front_curve]**2)
        opt_camber_curve[back_curve] = (opt_m_val / ((1.0 - opt_p_val)**2)) * ((1.0 - 2.0 * opt_p_val) + 2.0 * opt_p_val * chord_line_axis[back_curve] - chord_line_axis[back_curve]**2)
        
    ax.plot(chord_line_axis, opt_camber_curve, color='#00C6FF', linestyle=':', label='Optimized Camber Spine', alpha=0.7)
    ax.plot(optimized_coords[:, 0], optimized_coords[:, 1], color='#00C6FF', linewidth=2.5, label=f'Optimized Shape ({opt_naca_code})')
    ax.fill(optimized_coords[:, 0], optimized_coords[:, 1], color='#00C6FF', alpha=0.15)

ax.set_title("2D Airfoil Cross-Section", fontsize=14, fontweight='bold', color='#FFFFFF', pad=15)
ax.set_xlabel("Chord Station Space (x/c)", fontsize=11, color='#A1A8B8')
ax.set_ylabel("Camber / Thickness Scales (y/c)", fontsize=11, color='#A1A8B8')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.25, 0.25)
ax.set_aspect('equal', 'box')
ax.grid(True, color=(1, 1, 1, 0.08), linestyle='--')
ax.legend(facecolor='#1e2430', edgecolor=(1, 1, 1, 0.1), loc='upper right')

st.pyplot(fig)

st.markdown('<div class="section-title">📈 Performance Trends vs Angle of Attack</div>', unsafe_allow_html=True)

@st.cache_data
def run_cached_polar_sweep(camber, pos, thick, aspect_ratio, reynolds):
    aoa_sweep = np.linspace(0.0, 10.0, 30)
    cl_out, cd_out, eff_out = [], [], []
    
    for alpha_sweep_angle in aoa_sweep:
        sweep_run = analyze_airfoil(camber, pos, thick, alpha_sweep_angle, aspect_ratio, reynolds)
        cl_out.append(sweep_run['cl_nf'])
        cd_out.append(sweep_run['cd_nf'])
        eff_out.append(sweep_run['eff_nf'])
        
    return cl_out, cd_out, eff_out

aoa_sweep = np.linspace(0.0, 10.0, 30)

base_cl_array, base_cd_array, base_eff_array = run_cached_polar_sweep(
    camber_fraction, camber_pos_fraction, thickness_fraction, wing_ar, reynolds_selection
)

opt_cl_array, opt_cd_array, opt_eff_array = [], [], []
if st.session_state.opt_results:
    opt_results_data = st.session_state.opt_results
    opt_cl_array, opt_cd_array, opt_eff_array = run_cached_polar_sweep(
        opt_results_data['opt_m'], opt_results_data['opt_p'], opt_results_data['opt_t'], wing_ar, reynolds_selection
    )

graph_col1, graph_col2 = st.columns(2)

with graph_col1:
    fig_lift, ax_lift = plt.subplots(figsize=(6, 4))
    fig_lift.patch.set_facecolor('#0E1117')
    ax_lift.set_facecolor('#1e2430')
    
    ax_lift.plot(aoa_sweep, base_cl_array, color='#FF4B4B', linewidth=2, label='Original Profile')
    if st.session_state.opt_results:
        opt_results_data = st.session_state.opt_results
        ax_lift.plot(aoa_sweep, opt_cl_array, color='#00C6FF', linewidth=2, label='Optimized Profile')
        ax_lift.axvline(opt_results_data['opt_alpha'], color='#00C6FF', linestyle='--', alpha=0.6, label=f'Optimal Pitch ({opt_results_data["opt_alpha"]:.1f}°)')
    ax_lift.axvline(aoa_deg, color='#FF4B4B', linestyle='--', alpha=0.6, label=f'Original Pitch ({aoa_deg:.1f}°)')
    
    ax_lift.set_title("Lift Coefficient ($C_L$) vs AoA", fontsize=11, fontweight='bold')
    ax_lift.set_xlabel("AoA (deg)", fontsize=9, color='#A1A8B8')
    ax_lift.set_ylabel("$C_L$", fontsize=9, color='#A1A8B8')
    ax_lift.grid(True, color=(1, 1, 1, 0.08), linestyle='--')
    ax_lift.legend(facecolor='#1e2430', edgecolor=(1, 1, 1, 0.1), fontsize=8)
    st.pyplot(fig_lift)

with graph_col2:
    fig_eff, ax_eff = plt.subplots(figsize=(6, 4))
    fig_eff.patch.set_facecolor('#0E1117')
    ax_eff.set_facecolor('#1e2430')
    
    ax_eff.plot(aoa_sweep, base_eff_array, color='#FF4B4B', linewidth=2, label='Original Profile')
    if st.session_state.opt_results:
        opt_results_data = st.session_state.opt_results
        ax_eff.plot(aoa_sweep, opt_eff_array, color='#00C6FF', linewidth=2, label='Optimized Profile')
        ax_eff.axvline(opt_results_data['opt_alpha'], color='#00C6FF', linestyle='--', alpha=0.6)
    ax_eff.axvline(aoa_deg, color='#FF4B4B', linestyle='--', alpha=0.6)
    
    ax_eff.set_title("Total Efficiency ($C_L/C_D$) vs AoA", fontsize=11, fontweight='bold')
    ax_eff.set_xlabel("AoA (deg)", fontsize=9, color='#A1A8B8')
    ax_eff.set_ylabel("Airfoil Efficiency ($C_L/C_D$)", fontsize=9, color='#A1A8B8')
    ax_eff.grid(True, color=(1, 1, 1, 0.08), linestyle='--')
    ax_eff.legend(facecolor='#1e2430', edgecolor=(1, 1, 1, 0.1), fontsize=8)
    st.pyplot(fig_eff)

st.info("💡 **Dashboard Summary**: This playground cross-references frictionless fluid math with physics-informed Neural Network predictions (NeuralFoil) to balance swift performance calculations with realistic viscous profile drag tracking.")
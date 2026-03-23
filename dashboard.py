#
#
# # dashboard.py
# """
# Smart Truck Routing — Phase 3 & Phase 4 (ETA, Route Scoring, Re-routing)
# Full integrated Streamlit app (final version for your project).
# Save as dashboard.py and run:
#     streamlit run dashboard.py
# """
#
# import os
# from typing import Tuple
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score
#
# # -------------
# # Page config & visuals
# # -------------
# sns.set(style="whitegrid")
# st.set_page_config(page_title="Smart Truck Routing", layout="wide", page_icon="🚛")
#
# # --- Constants ---
# DATA_PATH = "clean_trips.csv"  # change if your file has a different name
# FUEL_PRICE = 95.0  # INR per litre (used for fuel cost estimation)
#
# # --------------------------
# # Utility / helper functions
# # --------------------------
# @st.cache_data
# def load_data(path: str = DATA_PATH) -> pd.DataFrame:
#     """Load CSV and do minimal normalization of column names."""
#     if not os.path.exists(path):
#         st.error(f"Data file not found: {path}. Place your cleaned CSV in the app folder or edit DATA_PATH.")
#         return pd.DataFrame()
#     df = pd.read_csv(path)
#     # normalize column names
#     df.columns = [c.strip() for c in df.columns]
#     return df
#
#
# def safe_map(series: pd.Series, mapping: dict, default=0):
#     """Map values safely using mapping dict and fill missing with default."""
#     return series.map(mapping).fillna(default)
#
#
# def estimate_toll(distance_km: float) -> float:
#     """Simple estimated toll cost by distance bucket."""
#     if distance_km < 100:
#         return 100.0
#     elif distance_km < 300:
#         return 250.0
#     else:
#         return 500.0
#
#
# def ensure_columns_for_model(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Ensure the dataframe has the columns required by ETA model and scoring:
#       - adjusted_speed
#       - weather_factor (numeric)
#       - traffic_factor (numeric)
#       - delay_min
#       - fuel_efficiency_kmpl
#       - travel_time_hr
#     This function modifies df in-place (returns it for chaining).
#     """
#     # travel_time_hr: use existing column or compute from actual_duration_min if available
#     if "travel_time_hr" not in df.columns:
#         if "actual_duration_min" in df.columns:
#             df["travel_time_hr"] = df["actual_duration_min"] / 60.0
#         elif "Scheduled_Arrival" in df.columns and "Scheduled_Departure" in df.columns:
#             # try parsing datetimes if present
#             try:
#                 df["Scheduled_Departure"] = pd.to_datetime(df["Scheduled_Departure"])
#                 df["Scheduled_Arrival"] = pd.to_datetime(df["Scheduled_Arrival"])
#                 df["travel_time_hr"] = (df["Scheduled_Arrival"] - df["Scheduled_Departure"]).dt.total_seconds() / 3600.0
#             except Exception:
#                 df["travel_time_hr"] = df.get("Distance_km", 0.0) / 50.0  # fallback guess
#         else:
#             # fallback: estimate travel hr from distance assuming 50 km/h
#             df["travel_time_hr"] = df.get("Distance_km", 0.0) / 50.0
#
#     # computed_speed: distance / actual duration (km/h) if available
#     if "computed_speed" not in df.columns:
#         # avoid division by zero
#         if "actual_duration_min" in df.columns and (df["actual_duration_min"] > 0).any():
#             df["computed_speed"] = df["Distance_km"] / (df["actual_duration_min"] / 60.0 + 1e-6)
#         elif (df["travel_time_hr"] > 0).any():
#             df["computed_speed"] = df["Distance_km"] / (df["travel_time_hr"] + 1e-6)
#         else:
#             df["computed_speed"] = 50.0  # safe default
#
#     # delay_min: unify names
#     if "delay_min" not in df.columns:
#         if "Delay_Minutes" in df.columns:
#             df["delay_min"] = df["Delay_Minutes"]
#         else:
#             df["delay_min"] = df.get("delay_min", 0.0)
#     df["delay_min"] = df["delay_min"].fillna(df["delay_min"].median())
#
#     # weather_factor: map textual weather to numeric penalty factor
#     wf_map = {"Clear": 1.0, "Cloudy": 1.05, "Rain": 1.15, "Fog": 1.25, "Snow": 1.4}
#     if "Weather_Condition" in df.columns:
#         df["weather_factor"] = df["Weather_Condition"].map(wf_map).fillna(1.1)
#     else:
#         df["weather_factor"] = 1.1
#
#     # traffic_factor: unify possible column names and map to numeric
#     traffic_map = {"Low": 1.0, "Light": 1.0, "Medium": 1.1, "Heavy": 1.25, "High": 1.25}
#     if "Traffic_Condition" in df.columns:
#         df["traffic_factor"] = df["Traffic_Condition"].map(traffic_map).fillna(1.1)
#     elif "Traffic_Level" in df.columns:
#         df["traffic_factor"] = df["Traffic_Level"].map(traffic_map).fillna(1.1)
#     elif "Traffic" in df.columns:
#         df["traffic_factor"] = df["Traffic"].map(traffic_map).fillna(1.1)
#     else:
#         df["traffic_factor"] = 1.1
#
#     # fuel efficiency: try to derive from Distance & Fuel_Consumption_L
#     if "fuel_efficiency_kmpl" not in df.columns:
#         if "Fuel_Consumption_L" in df.columns and "Distance_km" in df.columns:
#             df["fuel_efficiency_kmpl"] = df["Distance_km"] / (df["Fuel_Consumption_L"] + 1e-6)
#         else:
#             df["fuel_efficiency_kmpl"] = df.get("fuel_efficiency_kmpl", 3.7)  # default
#
#     # speed limit simulation & adjusted_speed
#     if "speed_limit" not in df.columns:
#         df["road_type"] = np.where(df["Distance_km"] > 200, "Highway", "City")
#         speed_limits = {"Highway": 80.0, "City": 50.0, "Rural": 60.0}
#         df["speed_limit"] = df["road_type"].map(speed_limits).fillna(60.0)
#     # replace infinite and NaNs
#     df["computed_speed"] = df["computed_speed"].replace([np.inf, -np.inf], np.nan).fillna(50.0)
#     df["adjusted_speed"] = np.minimum(df["computed_speed"], df["speed_limit"]).fillna(50.0)
#
#     # travel_time_hr (recompute safe)
#     df["travel_time_hr"] = df["Distance_km"] / (df["adjusted_speed"] + 1e-6)
#
#     # toll_cost and toll_delay if missing
#     if "toll_cost" not in df.columns:
#         df["toll_cost"] = df["Distance_km"].apply(estimate_toll)
#     if "toll_delay" not in df.columns:
#         df["toll_delay"] = df["Distance_km"].apply(lambda d: 0.1 if d < 100 else (0.2 if d < 300 else 0.3))
#
#     return df
#
#
# # ---------------------------------------
# # Baseline ETA Model (Linear Regression)
# # ---------------------------------------
# @st.cache_resource
# def train_eta_model(df: pd.DataFrame) -> Tuple[LinearRegression, dict]:
#     """
#     Train a baseline ETA model. Uses safe defaults if columns missing.
#     Returns (model, diagnostics)
#     diagnostics contains: mae, r2, features, X_test, y_test, y_pred
#     """
#     d = df.copy()
#     # features we want to use - ensure columns exist
#     features = ["Distance_km", "adjusted_speed", "weather_factor", "traffic_factor", "delay_min", "toll_cost"]
#     for f in features:
#         if f not in d.columns:
#             if f == "adjusted_speed":
#                 d[f] = np.minimum(d.get("computed_speed", 50.0), d.get("speed_limit", 60.0)).fillna(50.0)
#             else:
#                 d[f] = d.get(f, 0.0)
#
#     # target: if ETA_hr exists (observed), use it; else use actual_duration_min or construct proxy
#     if "ETA_hr" not in d.columns:
#         if "actual_duration_min" in d.columns:
#             d["ETA_hr"] = d["actual_duration_min"] / 60.0
#         else:
#             # construct synthetic ETA target from base travel time + penalties
#             d["ETA_hr"] = (
#                 d["travel_time_hr"]
#                 + d["travel_time_hr"] * (d["weather_factor"] - 1.0)
#                 + d["travel_time_hr"] * (d["traffic_factor"] - 1.0)
#                 + d["delay_min"] / 60.0
#                 + d["toll_delay"]
#             )
#
#     # drop rows with NaN in features or target
#     d = d.dropna(subset=features + ["ETA_hr"])
#     if d.shape[0] < 50:
#         # not enough data to train - return a dummy model with safe diagnostics
#         dummy_model = LinearRegression()
#         diagnostics = {"mae": None, "r2": None, "features": features, "X_test": None, "y_test": None, "y_pred": None}
#         return dummy_model, diagnostics
#
#     X = d[features]
#     y = d["ETA_hr"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     diagnostics = {"mae": mae, "r2": r2, "features": features, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}
#     return model, diagnostics
#
#
# # ---------------------------------------
# # Phase 4: Re-routing logic (rule-based)
# # ---------------------------------------
# def get_rerouted_path(current_route: str, weather: str, traffic_level: str, delay_min: float) -> Tuple[str, str]:
#     """
#     Very simple rule-based rerouting.
#     Returns (new_route_id, reason) - if new_route == current_route, no reroute.
#     """
#     reroute_reason = None
#     new_route = current_route
#
#     # priority: extreme weather > extreme delay > heavy traffic
#     if weather in ["Snow", "Storm", "Fog"]:
#         new_route = f"{current_route}_ALT_WEATHER"
#         reroute_reason = f"Severe weather: {weather}"
#
#     if delay_min > 90:  # more than 1.5 hours delay
#         new_route = f"{current_route}_ALT_DELAY"
#         reroute_reason = f"High accumulated delay: {delay_min:.0f} min"
#
#     if traffic_level in ["High", "Heavy"]:
#         # if route not yet changed by weather/delay, change to traffic alt
#         if new_route == current_route:
#             new_route = f"{current_route}_ALT_TRAFFIC"
#             reroute_reason = f"Heavy traffic: {traffic_level}"
#
#     return new_route, reroute_reason
#
#
# # -----------------------
# # Main App
# # -----------------------
# def main():
#     st.title("🚚 Smart Truck Routing — ETA, Scoring & Re-routing")
#     st.markdown(
#         "Interactive dashboard (Phase 3 & 4). Upload your cleaned CSV (`clean_trips.csv`) "
#         "or place it in the app folder. The app computes baseline ETA, ranks routes, "
#         "and offers a rule-based re-routing recommendation."
#     )
#
#     # Load
#     df = load_data(DATA_PATH)
#     if df.empty:
#         return
#
#     # Ensure required columns present and derived values are available
#     df = ensure_columns_for_model(df)
#
#     # Show a dataset summary
#     with st.expander("Dataset preview & diagnostics", expanded=False):
#         st.write("Rows:", df.shape[0], "Columns:", df.shape[1])
#         st.dataframe(df.head(8))
#         st.write(df.describe(include="all").T)
#
#     # Sidebar filters
#     st.sidebar.header("Filters & Controls")
#
#     start_options = ["All"] + sorted(df["Start_Location"].dropna().unique().tolist()) if "Start_Location" in df.columns else ["All"]
#     end_options = ["All"] + sorted(df["End_Location"].dropna().unique().tolist()) if "End_Location" in df.columns else ["All"]
#
#     start_loc = st.sidebar.selectbox("Start Location", options=start_options)
#     end_loc = st.sidebar.selectbox("End Location", options=end_options)
#
#     weather_opts = sorted(df.get("Weather_Condition", pd.Series()).dropna().unique().tolist())
#     weather_filter = st.sidebar.multiselect("Weather", options=weather_opts)
#
#     traffic = st.sidebar.selectbox(
#         "Traffic",
#         ["Low", "Medium", "High", "Jam"]
#     )
#
#     limit_rows = st.sidebar.slider("Plot sample size (for performance)", min_value=500, max_value=min(5000, len(df)), value=2000, step=500)
#
#     # apply filters
#     dff = df.copy()
#     if start_loc != "All":
#         dff = dff[dff["Start_Location"] == start_loc]
#     if end_loc != "All":
#         dff = dff[dff["End_Location"] == end_loc]
#     if weather_filter:
#         dff = dff[dff["Weather_Condition"].isin(weather_filter)]
#
#
#
#     st.markdown(f"**Filtered rows:** {dff.shape[0]}")
#
#
#     # Train ETA model (cached)
#     with st.spinner("Training baseline ETA model..."):
#         model, diag = train_eta_model(df)
#
#     # show diagnostics in sidebar
#     st.sidebar.subheader("Model diagnostics (baseline ETA)")
#     if diag.get("r2") is not None:
#         st.sidebar.write(f"R²: {diag['r2']:.3f}")
#         st.sidebar.write(f"MAE (hrs): {diag['mae']:.3f}")
#     else:
#         st.sidebar.write("Not enough data to compute diagnostics.")
#
#     # Phase 4 - Re-routing inputs
#     st.sidebar.subheader("Re-routing System (Phase 4)")
#     current_route_id = st.sidebar.text_input("Current Route ID", value="R1000")
#     # small normalized traffic choices
#     reroute_traffic = st.sidebar.selectbox("Traffic Level now", options=["Low", "Medium", "High"])
#     reroute_weather = st.sidebar.selectbox("Weather now", options=["Clear", "Cloudy", "Rain", "Fog", "Storm", "Snow"])
#     reroute_delay = st.sidebar.number_input("Accumulated delay (min)", min_value=0, value=0)
#
#     # -----------------------
#     # Compute composite route score on filtered df
#     # -----------------------
#     # Ensure the columns used exist
#     if "travel_time_hr" not in dff.columns:
#         dff["travel_time_hr"] = dff["Distance_km"] / (dff["adjusted_speed"] + 1e-6)
#     if "fuel_efficiency_kmpl" not in dff.columns:
#         dff["fuel_efficiency_kmpl"] = dff.get("fuel_efficiency_kmpl", 3.7)
#
#     # compute fuel cost (INR)
#     dff["fuel_cost"] = (dff["Distance_km"] / (dff["fuel_efficiency_kmpl"] + 1e-9)) * FUEL_PRICE
#     # weather penalty uses weather_factor (1.0 -> no penalty)
#     dff["weather_penalty"] = dff["travel_time_hr"] * (dff["weather_factor"] - 1.0)
#     # delay penalty: scale to hours
#     dff["delay_penalty"] = (dff["delay_min"] / 60.0) * 0.5
#     # toll delay is already estimated in hours proxy
#     if "toll_delay" not in dff.columns:
#         dff["toll_delay"] = dff["Distance_km"].apply(lambda d: 0.1 if d < 100 else (0.2 if d < 300 else 0.3))
#
#     dff["route_score"] = (
#         dff["travel_time_hr"]
#         + dff["weather_penalty"]
#         + dff["delay_penalty"]
#         + (dff["fuel_cost"] / 1000.0)
#         + dff["toll_delay"]
#     )
#     dff["route_rank"] = dff["route_score"].rank(method="dense")
#
#     # Top routes display
#     st.subheader("Top Recommended Routes (by composite score)")
#     top_routes = dff.sort_values("route_score").head(10).reset_index(drop=True)
#     display_cols = [
#         c for c in ["Trip_ID", "Start_Location", "End_Location", "Distance_km", "travel_time_hr",
#                     "fuel_cost", "delay_min", "Weather_Condition", "route_score", "route_rank"] if c in top_routes.columns
#     ]
#     st.dataframe(top_routes[display_cols].head(10), use_container_width=True)
#
#     # KPIs
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Filtered Trips", dff.shape[0])
#     col2.metric("Avg Delay (min)", f"{dff['delay_min'].mean():.1f}")
#     col3.metric("Avg Fuel Efficiency (km/l)", f"{dff['fuel_efficiency_kmpl'].mean():.2f}")
#
#     # Route score breakdown (top 5)
#     st.subheader("Route Score Breakdown (Top 5)")
#     top5 = top_routes.head(5).copy()
#     if not top5.empty:
#         fig, ax = plt.subplots(figsize=(10, 5))
#         x = np.arange(len(top5))
#         travel_time = top5["travel_time_hr"].values
#         weather_penalty = top5["weather_penalty"].values
#         delay_penalty = top5["delay_penalty"].values
#         fuel_penalty = (top5["fuel_cost"] / 1000.0).values
#         toll_penalty = top5["toll_delay"].values
#
#         bottom = np.zeros_like(travel_time)
#         ax.bar(x, travel_time, label="Travel Time (hr)")
#         bottom = bottom + travel_time
#         ax.bar(x, weather_penalty, bottom=bottom, label="Weather Penalty")
#         bottom = bottom + weather_penalty
#         ax.bar(x, delay_penalty, bottom=bottom, label="Delay Penalty")
#         bottom = bottom + delay_penalty
#         ax.bar(x, fuel_penalty, bottom=bottom, label="Fuel Penalty (x/1000)")
#         bottom = bottom + fuel_penalty
#         ax.bar(x, toll_penalty, bottom=bottom, label="Toll Delay (hr)")
#         ax.set_xticks(x)
#         ax.set_xticklabels(top5["Trip_ID"].astype(str), rotation=45)
#         ax.set_ylabel("Composite components (hours)")
#         ax.legend()
#         st.pyplot(fig)
#
#     # Model validation scatter
#     st.subheader("Model: Predicted vs Actual (sample)")
#     if diag.get("X_test") is not None:
#         y_test = diag["y_test"]
#         y_pred = diag["y_pred"]
#         fig2, ax2 = plt.subplots(figsize=(6, 6))
#         ax2.scatter(y_test, y_pred, alpha=0.6)
#         ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#         ax2.set_xlabel("Actual ETA (hr)")
#         ax2.set_ylabel("Predicted ETA (hr)")
#         ax2.set_title(f"Predicted vs Actual (R2={diag['r2']:.3f})")
#         st.pyplot(fig2)
#         st.write("Model MAE (hrs):", f"{diag['mae']:.3f}")
#     else:
#         st.info("Not enough data to show model validation (small dataset).")
#
#     # ETA prediction small form
#     st.sidebar.subheader("Predict ETA for a new trip")
#     with st.sidebar.form("eta_form"):
#         in_distance = st.number_input("Distance (km)", min_value=1.0, value=100.0)
#         in_speed = st.number_input("Estimated Speed (km/h)", min_value=10.0, value=70.0)
#         in_weather = st.selectbox("Weather", options=["Clear", "Cloudy", "Rain", "Fog", "Snow"])
#         in_traffic = st.selectbox("Traffic", options=["Low", "Medium", "High"])
#         in_delay = st.number_input("Known delay (min)", min_value=0, value=0)
#         submitted = st.form_submit_button("Predict ETA")
#         if submitted:
#             road_type = "Highway" if in_distance > 200 else "City"
#             speed_limit = 80 if road_type == "Highway" else 50
#             adjusted_speed = min(in_speed, speed_limit)
#             base_travel_hr = in_distance / (adjusted_speed + 1e-6)
#             wf_map = {"Clear": 1.0, "Cloudy": 1.05, "Rain": 1.15, "Fog": 1.25, "Snow": 1.4}
#             tf_map = {"Low": 1.0, "Medium": 1.1, "High": 1.25}
#             weather_factor_val = wf_map[in_weather]
#             traffic_factor_val = tf_map[in_traffic]
#             toll_cost_est = estimate_toll(in_distance)
#             X_new = pd.DataFrame([{
#                 "Distance_km": in_distance,
#                 "adjusted_speed": adjusted_speed,
#                 "weather_factor": weather_factor_val,
#                 "traffic_factor": traffic_factor_val,
#                 "delay_min": in_delay,
#                 "toll_cost": toll_cost_est
#             }])
#             try:
#                 pred_eta = model.predict(X_new)[0]
#             except Exception:
#                 pred_eta = base_travel_hr + (in_delay / 60.0) + base_travel_hr * (weather_factor_val - 1.0) + base_travel_hr * (traffic_factor_val - 1.0)
#             st.sidebar.success(f"Predicted ETA: {pred_eta:.2f} hours ({pred_eta * 60:.0f} minutes)")
#             st.sidebar.write(f"Base travel time (hr): {base_travel_hr:.2f}")
#             st.sidebar.write(f"Adjusted speed used: {adjusted_speed} km/h")
#             st.sidebar.write(f"Estimated toll cost: ₹{toll_cost_est}")
#
#     # Export top routes
#     if st.button("Export Top Routes to CSV"):
#         out_path = "top_routes_export.csv"
#         top_routes.to_csv(out_path, index=False)
#         st.success(f"Top routes saved to {out_path}")
#
#     # Phase 4: Re-routing check (using current sidebar inputs)
#     if st.sidebar.button("Check Re-routing Now"):
#         new_route, reason = get_rerouted_path(current_route_id, reroute_weather, reroute_traffic, reroute_delay)
#         st.markdown("### 🔄 Re-routing Recommendation")
#         if new_route != current_route_id and reason is not None:
#             st.error(f"⚠️ {reason}")
#             st.success(f"**Suggested new route:** {new_route}")
#         else:
#             st.info("✅ No re-routing recommended - current route remains optimal.")
#
#     # -----------------------
#     # 🔍 DEBUG PANEL — SHOW RULE APPLIED
#     # -----------------------
#     st.subheader("🔍 Re-routing Debug Panel")
#
#     # Get new route & reason
#     new_route, reason = get_rerouted_path(
#         current_route_id,
#         reroute_weather,
#         reroute_traffic,
#         reroute_delay
#     )
#
#     # Display inputs
#     st.write(f"**Current Route:** {current_route_id}")
#     st.write(f"**Weather Now:** {reroute_weather}")
#     st.write(f"**Traffic Now:** {reroute_traffic}")
#     st.write(f"**Accumulated Delay:** {reroute_delay} min")
#
#     # Show re-routing output
#     if reason:
#         st.success(
#             f"### ✔ Re-routing Activated!\n"
#             f"**New Route:** {new_route}\n\n"
#             f"**Reason:** {reason}"
#         )
#     else:
#         st.info("ℹ No re-routing applied. Conditions are normal.")
#
#     # Footer
#     st.markdown("---")
#     st.caption("Smart Truck Routing — Prototype. Next steps: integrate live APIs (Google Maps, OpenWeather), add XGBoost model for ETA, and deploy.")
#
# if __name__ == "__main__":
#     main()
# # dashboard.py
# """
# Smart Truck Routing — Phase 3 & Phase 4 (Upgraded)
# Features:
#  - ETA baseline + RandomForest explanation
#  - Composite route scoring
#  - Fuel cost predictor (diesel price slider)
#  - Route stability metric (delay variance)
#  - Best departure time recommendation (hour-of-day)
#  - Map visualization (Folium, optional)
#  - Route timeline simulation (static)
#  - Driver safety alerts
#  - Compare top routes
#  - Debug reroute panel
# Run:
#     streamlit run dashboard.py
# """
#
# import os
# from typing import Tuple, Optional
# import math
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # basic ML
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score
#
# # Optional imports (guarded)
# try:
#     from sklearn.ensemble import RandomForestRegressor
#     RF_AVAILABLE = True
# except Exception:
#     RF_AVAILABLE = False
#
# try:
#     import requests
#     REQ_AVAILABLE = True
# except Exception:
#     REQ_AVAILABLE = False
#
# try:
#     import folium
#     from streamlit_folium import st_folium
#     FOLIUM_AVAILABLE = True
# except Exception:
#     FOLIUM_AVAILABLE = False
#
# try:
#     import shap
#     SHAP_AVAILABLE = True
# except Exception:
#     SHAP_AVAILABLE = False
#
# # visuals
# sns.set(style="whitegrid")
# st.set_page_config(page_title="Smart Truck Routing — Advanced", layout="wide", page_icon="🚛")
#
# # ---------------- Constants ----------------
# DATA_PATH = "clean_trips.csv"
# DEFAULT_DIESEL = 95.0  # ₹ / litre
#
# # --------------- Helpers & Ensuring Columns ---------------
# @st.cache_data
# def load_data(path: str = DATA_PATH) -> pd.DataFrame:
#     if not os.path.exists(path):
#         st.error(f"Data not found: {path}. Place CSV in app folder or change DATA_PATH.")
#         return pd.DataFrame()
#     df = pd.read_csv(path)
#     df.columns = [c.strip() for c in df.columns]
#     return df
#
# def estimate_toll(distance_km: float) -> float:
#     if distance_km < 100:
#         return 100.0
#     elif distance_km < 300:
#         return 250.0
#     else:
#         return 500.0
#
# def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
#     # travel_time_hr
#     if "travel_time_hr" not in df.columns:
#         if "actual_duration_min" in df.columns:
#             df["travel_time_hr"] = df["actual_duration_min"] / 60.0
#         else:
#             df["travel_time_hr"] = df.get("Distance_km", 0.0) / 50.0
#     # computed_speed
#     if "computed_speed" not in df.columns:
#         df["computed_speed"] = df["Distance_km"] / (df["travel_time_hr"] + 1e-6)
#     df["computed_speed"] = df["computed_speed"].replace([np.inf, -np.inf], np.nan).fillna(50.0)
#     # delay
#     if "delay_min" not in df.columns:
#         if "Delay_Minutes" in df.columns:
#             df["delay_min"] = df["Delay_Minutes"]
#         else:
#             df["delay_min"] = 0.0
#     df["delay_min"] = df["delay_min"].fillna(df["delay_min"].median())
#     # weather factor
#     wf_map = {"Clear":1.0, "Cloudy":1.05, "Rain":1.15, "Fog":1.25, "Snow":1.4, "Storm":1.5}
#     if "Weather_Condition" in df.columns:
#         df["weather_factor"] = df["Weather_Condition"].map(wf_map).fillna(1.1)
#     else:
#         df["weather_factor"] = 1.1
#     # traffic
#     traffic_map = {"Low":1.0, "Light":1.0, "Medium":1.1, "Heavy":1.25, "High":1.25, "Jam":1.4}
#     if "Traffic_Condition" in df.columns:
#         df["traffic_factor"] = df["Traffic_Condition"].map(traffic_map).fillna(1.1)
#     elif "Traffic_Level" in df.columns:
#         df["traffic_factor"] = df["Traffic_Level"].map(traffic_map).fillna(1.1)
#     else:
#         # leave default and optionally create synthetic later
#         df["traffic_factor"] = 1.1
#     # fuel efficiency
#     if "fuel_efficiency_kmpl" not in df.columns:
#         if "Fuel_Consumption_L" in df.columns:
#             df["fuel_efficiency_kmpl"] = df["Distance_km"] / (df["Fuel_Consumption_L"] + 1e-6)
#         else:
#             df["fuel_efficiency_kmpl"] = 3.7
#     # speed limits & adjusted speed
#     if "speed_limit" not in df.columns:
#         df["road_type"] = np.where(df["Distance_km"] > 200, "Highway", "City")
#         df["speed_limit"] = df["road_type"].map({"Highway":80.0, "City":50.0, "Rural":60.0}).fillna(60.0)
#     df["adjusted_speed"] = np.minimum(df["computed_speed"], df["speed_limit"]).fillna(50.0)
#     df["travel_time_hr"] = df["Distance_km"] / (df["adjusted_speed"] + 1e-6)
#     # tolls
#     if "toll_cost" not in df.columns:
#         df["toll_cost"] = df["Distance_km"].apply(estimate_toll)
#     if "toll_delay" not in df.columns:
#         df["toll_delay"] = df["Distance_km"].apply(lambda d: 0.1 if d < 100 else (0.2 if d < 300 else 0.3))
#     # datetime extras (if scheduled timestamps exist)
#     for col in ["Scheduled_Departure","Scheduled_Arrival","Actual_Departure","Actual_Arrival"]:
#         if col in df.columns:
#             df[col] = pd.to_datetime(df[col], errors="coerce")
#     if "hour_of_day" not in df.columns:
#         if "Scheduled_Departure" in df.columns and df["Scheduled_Departure"].notna().any():
#             df["hour_of_day"] = df["Scheduled_Departure"].dt.hour.fillna(0).astype(int)
#         else:
#             # attempt from actual departure
#             if "Actual_Departure" in df.columns and df["Actual_Departure"].notna().any():
#                 df["hour_of_day"] = df["Actual_Departure"].dt.hour.fillna(0).astype(int)
#             else:
#                 df["hour_of_day"] = np.random.choice(range(24), size=len(df))
#     return df
#
# # -------------- Models & Diagnostics --------------
# @st.cache_resource
# def train_baseline_models(df: pd.DataFrame):
#     # Linear model (baseline)
#     features = ["Distance_km","adjusted_speed","weather_factor","traffic_factor","delay_min","toll_cost"]
#     d = df.copy()
#     d = d.dropna(subset=features, how="any")
#     if "ETA_hr" not in d.columns:
#         if "actual_duration_min" in d.columns:
#             d["ETA_hr"] = d["actual_duration_min"] / 60.0
#         else:
#             d["ETA_hr"] = d["travel_time_hr"] + d["travel_time_hr"]*(d["weather_factor"]-1.0) + d["travel_time_hr"]*(d["traffic_factor"]-1.0) + d["delay_min"]/60.0 + d["toll_delay"]
#     d = d.dropna(subset=["ETA_hr"], how="any")
#     diag = {}
#     # Linear
#     if d.shape[0] >= 30:
#         X = d[features]; y = d["ETA_hr"]
#         X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#         lin = LinearRegression().fit(X_train,y_train)
#         ypred = lin.predict(X_test)
#         diag["linear"] = {"model":lin,"mae":mean_absolute_error(y_test,ypred),"r2":r2_score(y_test,ypred)}
#     else:
#         diag["linear"] = None
#     # RandomForest for importance
#     if RF_AVAILABLE and d.shape[0] >= 50:
#         rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X,y)
#         importances = rf.feature_importances_
#         diag["rf"] = {"model":rf,"features":features,"importances":importances}
#     else:
#         diag["rf"] = None
#     return diag
#
# # --------------- Rerouting Logic (with debug) ---------------
# def get_rerouted_path(current_route: str, weather: str, traffic_level: str, delay_min: float) -> Tuple[str, Optional[str]]:
#     new_route = current_route
#     reason = None
#     w = (str(weather) or "").title()
#     t = (str(traffic_level) or "").title()
#     # priority: severe weather > delay > traffic
#     if w in ["Snow","Storm"]:
#         return f"{current_route}_ALT_WEATHER", f"Severe weather={w} -> choose weather-safe route"
#     if w == "Fog":
#         return f"{current_route}_ALT_FOG", f"Low visibility (Fog) -> prefer highways"
#     if delay_min > 90:
#         return f"{current_route}_ALT_DELAY", f"High accumulated delay={delay_min:.0f} min -> delay-avoidance route"
#     if t in ["Jam","High","Heavy"]:
#         return f"{current_route}_ALT_TRAFFIC", f"Heavy traffic={t} -> high-traffic diversion"
#     if t=="Medium" and w in ["Rain","Cloudy"]:
#         return f"{current_route}_ALT_MED", f"Medium traffic + {w} -> medium-safe route"
#     return new_route, None
#
# # --------------- Weather API helper (optional) ---------------
# def fetch_weather(lat:float, lon:float) -> Optional[dict]:
#     # Example: Open-Meteo (no key required)
#     if not REQ_AVAILABLE:
#         return None
#     try:
#         url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation"
#         resp = requests.get(url, timeout=6)
#         if resp.status_code==200:
#             return resp.json()
#     except Exception:
#         return None
#     return None
#
# # --------------- Route stability ---------------
# def compute_route_stability(df: pd.DataFrame, group_cols: list = ["Start_Location","End_Location"]):
#     # Stability: 1 - normalized variance of delay (lower variance -> higher stability)
#     g = df.groupby(group_cols)["delay_min"].agg(['mean','std','count']).reset_index()
#     g['std'] = g['std'].fillna(0.0)
#     # normalize std to [0,1] using quantiles
#     max_std = g['std'].quantile(0.99) or 1.0
#     g['stability'] = (1 - (g['std'] / (max_std+1e-6))).clip(0,1)*100
#     return g
#
# # --------------- Best departure time ---------------
# def best_departure_window(df: pd.DataFrame):
#     # compute avg delay per hour
#     if "hour_of_day" not in df.columns:
#         df["hour_of_day"] = np.random.choice(range(24), size=len(df))
#     agg = df.groupby("hour_of_day")["delay_min"].mean().reset_index()
#     best_hour = int(agg.loc[agg["delay_min"].idxmin(),"hour_of_day"])
#     # return window of +-1 hour
#     window = f"{best_hour}:00 - {(best_hour+1)%24}:00"
#     return window, agg
#
# # --------------- Route timeline (static simulation) ---------------
# def simulate_timeline(distance_km: float, avg_speed_kmph: float, stops: list = None):
#     # stops: list of (km_along_route, delay_min, label)
#     total_hours = distance_km / (avg_speed_kmph+1e-6)
#     times = np.linspace(0,total_hours,100)
#     dist = times * avg_speed_kmph
#     if stops is None:
#         stops = []
#     return times, dist, stops
#
# # --------------- Main App ---------------
# def main():
#     st.title("🚚 Smart Truck Routing — Advanced Prototype")
#     st.markdown("Phase 3/4 upgraded: ETA baseline, route scoring, rerouting, visualization, and explainability.")
#
#     # Load
#     df = load_data(DATA_PATH)
#     if df.empty:
#         return
#     df = ensure_columns(df)
#
#     # Top bar controls for global features
#     st.sidebar.header("Global Settings")
#     diesel_price = st.sidebar.slider("Diesel Price (₹/L)", min_value=70.0, max_value=140.0, value=float(DEFAULT_DIESEL), step=1.0)
#     show_map = st.sidebar.checkbox("Show Map (Folium)", value=FOLIUM_AVAILABLE)
#     show_xai = st.sidebar.checkbox("Show Model Explainability (Feature Importance)", value=RF_AVAILABLE)
#     # filters
#     st.sidebar.header("Filters")
#     start_loc = st.sidebar.selectbox("Start Location", options=["All"] + sorted(df["Start_Location"].dropna().unique().tolist()) if "Start_Location" in df.columns else ["All"])
#     end_loc = st.sidebar.selectbox("End Location", options=["All"] + sorted(df["End_Location"].dropna().unique().tolist()) if "End_Location" in df.columns else ["All"])
#     weather_filter = st.sidebar.multiselect("Weather", options=sorted(df.get("Weather_Condition", pd.Series()).dropna().unique().tolist()))
#     # rerouting inputs
#     st.sidebar.header("Re-routing Inputs")
#     current_route_id = st.sidebar.text_input("Current Route ID", value="R1000")
#     reroute_traffic = st.sidebar.selectbox("Traffic Level now", options=["Low","Medium","High","Jam"])
#     reroute_weather = st.sidebar.selectbox("Weather now", options=["Clear","Cloudy","Rain","Fog","Storm","Snow"])
#     reroute_delay = st.sidebar.number_input("Accumulated delay (min)", min_value=0, value=0)
#
#     # Apply filters
#     dff = df.copy()
#     if start_loc!="All":
#         dff = dff[dff["Start_Location"]==start_loc]
#     if end_loc!="All":
#         dff = dff[dff["End_Location"]==end_loc]
#     if weather_filter:
#         dff = dff[dff["Weather_Condition"].isin(weather_filter)]
#
#     st.markdown(f"**Filtered rows:** {dff.shape[0]}")
#
#     # Train baseline models (cached)
#     diag = train_baseline_models(df)
#
#     # Compute composite route score and ranking
#     dff["fuel_cost"] = (dff["Distance_km"] / (dff["fuel_efficiency_kmpl"]+1e-9)) * diesel_price
#     dff["weather_penalty"] = dff["travel_time_hr"] * (dff["weather_factor"] - 1.0)
#     dff["delay_penalty"] = (dff["delay_min"] / 60.0) * 0.5
#     dff["route_score"] = dff["travel_time_hr"] + dff["weather_penalty"] + dff["delay_penalty"] + (dff["fuel_cost"]/1000.0) + dff["toll_delay"]
#     dff["route_rank"] = dff["route_score"].rank(method="dense")
#     top_routes = dff.sort_values("route_score").head(10).reset_index(drop=True)
#
#     # Columns for display
#     display_cols = [c for c in ["Trip_ID","Start_Location","End_Location","Distance_km","travel_time_hr","fuel_cost","delay_min","Weather_Condition","route_score","route_rank"] if c in top_routes.columns]
#
#     # Layout: left = list, right = details
#     left, right = st.columns([1,2])
#     with left:
#         st.subheader("Top Recommended Routes")
#         st.dataframe(top_routes[display_cols].head(10), use_container_width=True)
#         if st.button("Export Top Routes CSV"):
#             out = "top_routes_export.csv"
#             top_routes.to_csv(out, index=False)
#             st.success(f"Saved {out}")
#
#         # KPIs
#         st.markdown("### KPIs")
#         st.metric("Filtered Trips", dff.shape[0])
#         st.metric("Avg Delay (min)", f"{dff['delay_min'].mean():.1f}")
#         st.metric("Avg Fuel Eff. (km/l)", f"{dff['fuel_efficiency_kmpl'].mean():.2f}")
#         # Best departure
#         window, agg_hour = best_departure_window(df)
#         st.markdown(f"**Best departure window (historical):** {window}")
#
#     with right:
#         st.subheader("Route Score Breakdown — Top 5")
#         top5 = top_routes.head(5).copy()
#         if not top5.empty:
#             fig, ax = plt.subplots(figsize=(10,4))
#             x = np.arange(len(top5))
#             tt = top5["travel_time_hr"].values
#             wp = top5["weather_penalty"].values
#             dp = top5["delay_penalty"].values
#             fp = (top5["fuel_cost"]/1000.0).values
#             tp = top5["toll_delay"].values
#             bottom = np.zeros_like(tt)
#             ax.bar(x, tt, label="Travel Time (hr)")
#             bottom += tt
#             ax.bar(x, wp, bottom=bottom, label="Weather Penalty")
#             bottom += wp
#             ax.bar(x, dp, bottom=bottom, label="Delay Penalty")
#             bottom += dp
#             ax.bar(x, fp, bottom=bottom, label="Fuel Penalty (x/1000)")
#             bottom += fp
#             ax.bar(x, tp, bottom=bottom, label="Toll Delay (hr)")
#             ax.set_xticks(x)
#             ax.set_xticklabels(top5["Trip_ID"].astype(str), rotation=45)
#             ax.set_ylabel("Composite (hours)")
#             ax.legend()
#             st.pyplot(fig)
#
#         # Compare top 2 routes (if exist)
#         st.markdown("## Compare Top 2 Routes")
#         if top_routes.shape[0] >= 2:
#             r1 = top_routes.iloc[0]
#             r2 = top_routes.iloc[1]
#             comp_df = pd.DataFrame({
#                 "Metric":["Distance_km","Travel_hr","Fuel Cost (₹)","Delay_min","Route Score"],
#                 "Route 1":[r1["Distance_km"], r1["travel_time_hr"], r1["fuel_cost"], r1["delay_min"], r1["route_score"]],
#                 "Route 2":[r2["Distance_km"], r2["travel_time_hr"], r2["fuel_cost"], r2["delay_min"], r2["route_score"]],
#                 "Better":[
#                     "1" if r1["Distance_km"]<=r2["Distance_km"] else "2",
#                     "1" if r1["travel_time_hr"]<=r2["travel_time_hr"] else "2",
#                     "1" if r1["fuel_cost"]<=r2["fuel_cost"] else "2",
#                     "1" if r1["delay_min"]<=r2["delay_min"] else "2",
#                     "1" if r1["route_score"]<=r2["route_score"] else "2",
#                 ]
#             })
#             st.table(comp_df)
#
#         # Feature importance (XAI)
#         st.markdown("## Model Explainability")
#         if diag.get("rf") is not None:
#             rf_info = diag["rf"]
#             feat = rf_info["features"]
#             imp = rf_info["importances"]
#             imp_df = pd.DataFrame({"feature":feat,"importance":imp}).sort_values("importance",ascending=False)
#             st.bar_chart(imp_df.set_index("feature")["importance"])
#             st.write("RandomForest feature importances (proxy for XAI).")
#             if SHAP_AVAILABLE:
#                 try:
#                     explainer = shap.TreeExplainer(rf_info["model"])
#                     sample = df[feat].sample(min(200, df.shape[0]))
#                     shap_vals = explainer.shap_values(sample)
#                     st.write("SHAP summary (sample) — requires SHAP in environment.")
#                     # shap plotting in streamlit is complex; show textual top 3 features
#                     mean_shap = np.abs(shap_vals).mean(axis=0)
#                     top3 = pd.Series(mean_shap, index=feat).sort_values(ascending=False).head(3)
#                     st.write("Top 3 SHAP features:", top3.to_dict())
#                 except Exception:
#                     st.write("SHAP explainability failed — ensure SHAP installed and working.")
#         else:
#             st.info("RandomForest not trained (insufficient data or sklearn missing) — feature importance unavailable.")
#
#         # Route stability
#         st.markdown("## Route Stability (historical)")
#         stability = compute_route_stability(df)
#         # find current start->end stability if available
#         if "Start_Location" in df.columns and "End_Location" in df.columns:
#             key_match = stability[
#                 (stability["Start_Location"]==top_routes.iloc[0]["Start_Location"]) &
#                 (stability["End_Location"]==top_routes.iloc[0]["End_Location"])
#             ]
#             if not key_match.empty:
#                 st.write(f"Stability for top route (0-100): {key_match.iloc[0]['stability']:.1f}%")
#         st.dataframe(stability.head(8))
#
#     # ----------------- Map view -----------------
#     st.markdown("## Map View (optional)")
#     if show_map and FOLIUM_AVAILABLE:
#         try:
#             m = folium.Map(location=[20.5937,78.9629], zoom_start=5)  # India center
#             # plot top routes origins/destinations (if lat/lon present)
#             # If lat/lon columns exist, plot them; else we skip
#             if "origin_lat" in df.columns and "origin_lon" in df.columns:
#                 for idx,row in top_routes.iterrows():
#                     folium.CircleMarker([row["origin_lat"],row["origin_lon"]], radius=4, popup=str(row["Trip_ID"])).add_to(m)
#             st_folium(m, width=700)
#         except Exception as e:
#             st.error("Map rendering failed: " + str(e))
#     else:
#         if show_map:
#             st.info("Folium or streamlit_folium not installed — map disabled.")
#
#     # ----------------- Timeline simulation for single route -----------------
#     st.markdown("## Route Timeline Simulation (static)")
#     if top_routes.shape[0] > 0:
#         sample = top_routes.iloc[0]
#         times, dist, stops = simulate_timeline(float(sample["Distance_km"]), float(sample["adjusted_speed"]), stops=[(20,10,"Toll1")])
#         fig3, ax3 = plt.subplots(figsize=(8,3))
#         ax3.plot(times*60, dist, label="Distance covered (km)")
#         ax3.set_xlabel("Time (minutes)")
#         ax3.set_ylabel("Distance (km)")
#         ax3.set_title(f"Simulated timeline for Trip {sample['Trip_ID']}")
#         st.pyplot(fig3)
#
#     # ----------------- Predict ETA small form + reroute suggestion -----------------
#     st.sidebar.markdown("---")
#     st.sidebar.subheader("Predict ETA for New Trip (with reroute)")
#     with st.sidebar.form("predict_form"):
#         in_distance = st.number_input("Distance (km)", min_value=1.0, value=100.0)
#         in_speed = st.number_input("Estimate Speed (km/h)", min_value=10.0, value=70.0)
#         in_weather = st.selectbox("Weather", ["Clear","Cloudy","Rain","Fog","Snow"])
#         in_traffic = st.selectbox("Traffic", ["Low","Medium","High"])
#         in_delay = st.number_input("Known delay (min)", min_value=0, value=0)
#         submit_pred = st.form_submit_button("Predict & Suggest")
#     if submit_pred:
#         road_type = "Highway" if in_distance>200 else "City"
#         speed_limit = 80 if road_type=="Highway" else 50
#         adjusted_speed = min(in_speed, speed_limit)
#         base_hr = in_distance/(adjusted_speed+1e-6)
#         wfmap = {"Clear":1.0,"Cloudy":1.05,"Rain":1.15,"Fog":1.25,"Snow":1.4}
#         tfmap = {"Low":1.0,"Medium":1.1,"High":1.25}
#         wf = wfmap[in_weather]; tf = tfmap[in_traffic]; toll = estimate_toll(in_distance)
#         Xnew = pd.DataFrame([{"Distance_km":in_distance,"adjusted_speed":adjusted_speed,"weather_factor":wf,"traffic_factor":tf,"delay_min":in_delay,"toll_cost":toll}])
#         pred = None
#         try:
#             if diag.get("rf") is not None:
#                 pred = diag["rf"]["model"].predict(Xnew)[0]
#             elif diag.get("linear") is not None:
#                 pred = diag["linear"]["model"].predict(Xnew)[0]
#         except Exception:
#             pred = base_hr + (in_delay/60.0) + base_hr*(wf-1.0) + base_hr*(tf-1.0)
#         st.sidebar.success(f"Pred ETA: {pred:.2f} hr ({pred*60:.0f} min)")
#         # reroute suggestion for hypothetical trip
#         suggested, reason = get_rerouted_path("NEW_TRIP", in_weather, in_traffic, in_delay)
#         if reason:
#             st.sidebar.error(f"Reroute suggested: {suggested}\nReason: {reason}")
#         else:
#             st.sidebar.info("No reroute for this hypothetical trip.")
#
#     # --------------- Debug reroute panel (explicit) ---------------
#     st.markdown("## 🔍 Re-routing Debug Panel")
#     st.write(f"Current route id (sidebar): **{current_route_id}**")
#     st.write(f"Traffic now: **{reroute_traffic}**, Weather now: **{reroute_weather}**, Delay: **{reroute_delay} min**")
#     if st.button("Check Re-routing Now"):
#         newr, reason = get_rerouted_path(current_route_id, reroute_weather, reroute_traffic, reroute_delay)
#         if reason:
#             st.error(f"⚠ {reason}")
#             st.success(f"Suggested new route: {newr}")
#         else:
#             st.info("No re-routing recommended.")
#
#     # --------------- Driver Safety Alert ---------------
#     st.markdown("## Safety Alerts")
#     alerts = []
#     # Severe weather
#     if reroute_weather in ["Fog","Storm","Snow"]:
#         alerts.append(f"Severe weather ({reroute_weather}). Recommend caution / reduce speed.")
#     if reroute_delay > 180:
#         alerts.append(f"Very high accumulated delay ({reroute_delay} min). Consider driver rest / reschedule.")
#     if len(alerts)>0:
#         for a in alerts:
#             st.warning(a)
#     else:
#         st.success("No active safety alerts.")
#
#     # --------------- Footer ---------------
#     st.markdown("---")
#     st.caption("Prototype — Next: integrate live Maps/Traffic APIs, production-grade routing solver, and deployment.")
#
# if __name__ == "__main__":
#     main()



#
# # dashboard.py
# """
# Smart Truck Routing — Phase 3 & Phase 4 (ETA, Route Scoring, Re-routing)
# Save as dashboard.py and run:
#     streamlit run dashboard.py
# """
#
# import os
# from typing import Tuple
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score
#
# # ------------- Page config & visuals -------------
# sns.set(style="whitegrid")
# st.set_page_config(page_title="Smart Truck Routing", layout="wide", page_icon="🚛")
#
# # --- Constants ---
# DATA_PATH = "clean_trips.csv"  # change if your file has a different name
# FUEL_PRICE = 95.0  # INR per litre (used for fuel cost estimation)
#
# # -------------------------- Helpers --------------------------
#
#
# @st.cache_data
# def load_data(path: str = DATA_PATH) -> pd.DataFrame:
#     """Load CSV and do minimal normalization of column names."""
#     if not os.path.exists(path):
#         st.error(f"Data file not found: {path}. Place your cleaned CSV in the app folder or edit DATA_PATH.")
#         return pd.DataFrame()
#     df = pd.read_csv(path)
#     df.columns = [c.strip() for c in df.columns]
#     return df
#
#
# def estimate_toll(distance_km: float) -> float:
#     if distance_km < 100:
#         return 100.0
#     elif distance_km < 300:
#         return 250.0
#     else:
#         return 500.0
#
#
# def ensure_columns_for_model(df: pd.DataFrame) -> pd.DataFrame:
#     """Ensure dataframe has required derived columns for scoring & model."""
#     # travel_time_hr
#     if "travel_time_hr" not in df.columns:
#         if "actual_duration_min" in df.columns:
#             df["travel_time_hr"] = df["actual_duration_min"] / 60.0
#         else:
#             df["travel_time_hr"] = df.get("Distance_km", 0.0) / 50.0
#
#     # computed_speed
#     if "computed_speed" not in df.columns:
#         if "actual_duration_min" in df.columns and (df["actual_duration_min"] > 0).any():
#             df["computed_speed"] = df["Distance_km"] / (df["actual_duration_min"] / 60.0 + 1e-6)
#         else:
#             df["computed_speed"] = df["Distance_km"] / (df["travel_time_hr"] + 1e-6)
#
#     # delay_min
#     if "delay_min" not in df.columns:
#         if "Delay_Minutes" in df.columns:
#             df["delay_min"] = df["Delay_Minutes"]
#         else:
#             df["delay_min"] = df.get("delay_min", 0.0)
#     df["delay_min"] = df["delay_min"].fillna(df["delay_min"].median())
#
#     # weather_factor
#     wf_map = {"Clear": 1.0, "Cloudy": 1.05, "Rain": 1.15, "Fog": 1.25, "Snow": 1.4, "Storm": 1.5}
#     if "Weather_Condition" in df.columns:
#         df["weather_factor"] = df["Weather_Condition"].map(wf_map).fillna(1.1)
#     else:
#         df["weather_factor"] = 1.1
#
#     # traffic_factor
#     traffic_map = {"Low": 1.0, "Light": 1.0, "Medium": 1.1, "Heavy": 1.25, "High": 1.25, "Jam": 1.4}
#     if "Traffic_Condition" in df.columns:
#         df["traffic_factor"] = df["Traffic_Condition"].map(traffic_map).fillna(1.1)
#     elif "Traffic_Level" in df.columns:
#         df["traffic_factor"] = df["Traffic_Level"].map(traffic_map).fillna(1.1)
#     elif "Traffic" in df.columns:
#         df["traffic_factor"] = df["Traffic"].map(traffic_map).fillna(1.1)
#     else:
#         df["traffic_factor"] = 1.1
#
#     # fuel_efficiency_kmpl
#     if "fuel_efficiency_kmpl" not in df.columns:
#         if "Fuel_Consumption_L" in df.columns and "Distance_km" in df.columns:
#             df["fuel_efficiency_kmpl"] = df["Distance_km"] / (df["Fuel_Consumption_L"] + 1e-6)
#         else:
#             df["fuel_efficiency_kmpl"] = df.get("fuel_efficiency_kmpl", 3.7)
#
#     # speed_limit and adjusted_speed
#     if "speed_limit" not in df.columns:
#         df["road_type"] = np.where(df["Distance_km"] > 200, "Highway", "City")
#         speed_limits = {"Highway": 80.0, "City": 50.0, "Rural": 60.0}
#         df["speed_limit"] = df["road_type"].map(speed_limits).fillna(60.0)
#
#     df["computed_speed"] = df["computed_speed"].replace([np.inf, -np.inf], np.nan).fillna(50.0)
#     df["adjusted_speed"] = np.minimum(df["computed_speed"], df["speed_limit"]).fillna(50.0)
#
#     # recompute travel_time_hr safely
#     df["travel_time_hr"] = df["Distance_km"] / (df["adjusted_speed"] + 1e-6)
#
#     # tolls
#     if "toll_cost" not in df.columns:
#         df["toll_cost"] = df["Distance_km"].apply(estimate_toll)
#     if "toll_delay" not in df.columns:
#         df["toll_delay"] = df["Distance_km"].apply(lambda d: 0.1 if d < 100 else (0.2 if d < 300 else 0.3))
#
#     return df
#
#
# @st.cache_resource
# def train_eta_model(df: pd.DataFrame) -> Tuple[LinearRegression, dict]:
#     """Train a baseline ETA linear model (safe: returns dummy model if not enough data)."""
#     d = df.copy()
#     features = ["Distance_km", "adjusted_speed", "weather_factor", "traffic_factor", "delay_min", "toll_cost"]
#     for f in features:
#         if f not in d.columns:
#             if f == "adjusted_speed":
#                 d[f] = np.minimum(d.get("computed_speed", 50.0), d.get("speed_limit", 60.0)).fillna(50.0)
#             else:
#                 d[f] = d.get(f, 0.0)
#
#     if "ETA_hr" not in d.columns:
#         if "actual_duration_min" in d.columns:
#             d["ETA_hr"] = d["actual_duration_min"] / 60.0
#         else:
#             d["ETA_hr"] = (
#                 d["travel_time_hr"]
#                 + d["travel_time_hr"] * (d["weather_factor"] - 1.0)
#                 + d["travel_time_hr"] * (d["traffic_factor"] - 1.0)
#                 + d["delay_min"] / 60.0
#                 + d["toll_delay"]
#             )
#
#     d = d.dropna(subset=features + ["ETA_hr"])
#     diagnostics = {"mae": None, "r2": None, "features": features, "X_test": None, "y_test": None, "y_pred": None}
#
#     if d.shape[0] < 50:
#         # Not enough data: return untrained model + diagnostics None
#         return LinearRegression(), diagnostics
#
#     X = d[features]
#     y = d["ETA_hr"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     diagnostics["mae"] = mean_absolute_error(y_test, y_pred)
#     diagnostics["r2"] = r2_score(y_test, y_pred)
#     diagnostics["features"] = features
#     diagnostics["X_test"] = X_test
#     diagnostics["y_test"] = y_test
#     diagnostics["y_pred"] = y_pred
#     return model, diagnostics
#
#
# # ---------------------------------------
# # Phase 4: Re-routing logic (rule-based)
# # ---------------------------------------
# def get_rerouted_path(current_route: str, weather: str, traffic_level: str, delay_min: float) -> Tuple[str, str]:
#     """
#     Simple rule-based rerouting.
#     Priority: extreme weather > extreme delay > heavy traffic.
#     Returns (new_route_id, reason)
#     """
#     reroute_reason = None
#     new_route = current_route
#
#     w = str(weather).title() if weather is not None else ""
#     t = str(traffic_level).title() if traffic_level is not None else ""
#
#     # severe weather
#     if w in ["Snow", "Storm"]:
#         return f"{current_route}_ALT_WEATHER", f"Severe weather: {w}"
#     if w == "Fog":
#         return f"{current_route}_ALT_FOG", "Low visibility (Fog) — prefer higher-speed/safer corridor"
#
#     # accumulated delay
#     if delay_min is not None and delay_min > 90:
#         return f"{current_route}_ALT_DELAY", f"High accumulated delay: {delay_min:.0f} min"
#
#     # heavy traffic
#     if t in ["Jam", "High", "Heavy"]:
#         return f"{current_route}_ALT_TRAFFIC", f"Heavy traffic: {t}"
#
#     # default: no reroute
#     return new_route, None
#
#
# # ----------------------- Main App -----------------------
# def main():
#     st.title("🚚 Smart Truck Routing — ETA, Scoring & Re-routing")
#     st.markdown(
#         "Interactive dashboard (Phase 3 & 4). Place your cleaned CSV (`clean_trips.csv`) in the app folder "
#         "or change DATA_PATH. App computes baseline ETA, ranks routes, and offers rule-based re-routing."
#     )
#
#     # Load
#     df = load_data(DATA_PATH)
#     if df.empty:
#         return
#
#     # Ensure derived columns
#     df = ensure_columns_for_model(df)
#
#     # Dataset preview
#     with st.expander("Dataset preview & diagnostics", expanded=False):
#         st.write("Rows:", df.shape[0], "Columns:", df.shape[1])
#         st.dataframe(df.head(8))
#         st.write(df.describe(include="all").T)
#
#     # Sidebar filters & reroute controls
#     st.sidebar.header("Filters & Controls")
#     start_options = ["All"] + sorted(df["Start_Location"].dropna().unique().tolist()) if "Start_Location" in df.columns else ["All"]
#     end_options = ["All"] + sorted(df["End_Location"].dropna().unique().tolist()) if "End_Location" in df.columns else ["All"]
#     start_loc = st.sidebar.selectbox("Start Location", options=start_options)
#     end_loc = st.sidebar.selectbox("End Location", options=end_options)
#     weather_opts = sorted(df.get("Weather_Condition", pd.Series()).dropna().unique().tolist())
#     weather_filter = st.sidebar.multiselect("Weather", options=weather_opts)
#
#     # Reroute inputs (explicit)
#     st.sidebar.subheader("Re-routing System (Phase 4)")
#     current_route_id = st.sidebar.text_input("Current Route ID", value="R1000")
#     reroute_traffic = st.sidebar.selectbox("Traffic Level now", options=["Low", "Medium", "High", "Jam"])
#     reroute_weather = st.sidebar.selectbox("Weather now", options=["Clear", "Cloudy", "Rain", "Fog", "Storm", "Snow"])
#     reroute_delay = st.sidebar.number_input("Accumulated delay (min)", min_value=0, value=0)
#
#     # plot sample slider
#     limit_rows = st.sidebar.slider("Plot sample size (for performance)", min_value=500, max_value=min(5000, len(df)), value=2000, step=500)
#
#     # apply filters
#     dff = df.copy()
#     if start_loc != "All":
#         dff = dff[dff["Start_Location"] == start_loc]
#     if end_loc != "All":
#         dff = dff[dff["End_Location"] == end_loc]
#     if weather_filter:
#         dff = dff[dff["Weather_Condition"].isin(weather_filter)]
#
#     st.markdown(f"**Filtered rows:** {dff.shape[0]}")
#
#     # Train ETA model (cached)
#     with st.spinner("Training baseline ETA model..."):
#         model, diag = train_eta_model(df)
#
#     # Model diagnostics in sidebar
#     st.sidebar.subheader("Model diagnostics (baseline ETA)")
#     if diag.get("r2") is not None:
#         st.sidebar.write(f"R²: {diag['r2']:.3f}")
#         st.sidebar.write(f"MAE (hrs): {diag['mae']:.3f}")
#     else:
#         st.sidebar.write("Not enough data to compute diagnostics (need >= 50 rows). Using heuristic fallback.")
#
#     # Composite route scoring
#     if "travel_time_hr" not in dff.columns:
#         dff["travel_time_hr"] = dff["Distance_km"] / (dff["adjusted_speed"] + 1e-6)
#     if "fuel_efficiency_kmpl" not in dff.columns:
#         dff["fuel_efficiency_kmpl"] = dff.get("fuel_efficiency_kmpl", 3.7)
#
#     dff["fuel_cost"] = (dff["Distance_km"] / (dff["fuel_efficiency_kmpl"] + 1e-9)) * FUEL_PRICE
#     dff["weather_penalty"] = dff["travel_time_hr"] * (dff["weather_factor"] - 1.0)
#     dff["delay_penalty"] = (dff["delay_min"] / 60.0) * 0.5
#     if "toll_delay" not in dff.columns:
#         dff["toll_delay"] = dff["Distance_km"].apply(lambda d: 0.1 if d < 100 else (0.2 if d < 300 else 0.3))
#
#     dff["route_score"] = (
#         dff["travel_time_hr"]
#         + dff["weather_penalty"]
#         + dff["delay_penalty"]
#         + (dff["fuel_cost"] / 1000.0)
#         + dff["toll_delay"]
#     )
#     dff["route_rank"] = dff["route_score"].rank(method="dense")
#
#     # Top routes
#     st.subheader("Top Recommended Routes (by composite score)")
#     top_routes = dff.sort_values("route_score").head(10).reset_index(drop=True)
#     display_cols = [
#         c for c in ["Trip_ID", "Start_Location", "End_Location", "Distance_km", "travel_time_hr",
#                     "fuel_cost", "delay_min", "Weather_Condition", "route_score", "route_rank"] if c in top_routes.columns
#     ]
#     st.dataframe(top_routes[display_cols].head(10), use_container_width=True)
#
#     # KPIs
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Filtered Trips", dff.shape[0])
#     col2.metric("Avg Delay (min)", f"{dff['delay_min'].mean():.1f}")
#     col3.metric("Avg Fuel Efficiency (km/l)", f"{dff['fuel_efficiency_kmpl'].mean():.2f}")
#
#     # Route score breakdown (top 5)
#     st.subheader("Route Score Breakdown (Top 5)")
#     top5 = top_routes.head(5).copy()
#     if not top5.empty:
#         fig, ax = plt.subplots(figsize=(10, 5))
#         x = np.arange(len(top5))
#         travel_time = top5["travel_time_hr"].values
#         weather_penalty = top5["weather_penalty"].values
#         delay_penalty = top5["delay_penalty"].values
#         fuel_penalty = (top5["fuel_cost"] / 1000.0).values
#         toll_penalty = top5["toll_delay"].values
#         bottom = np.zeros_like(travel_time)
#         ax.bar(x, travel_time, label="Travel Time (hr)")
#         bottom = bottom + travel_time
#         ax.bar(x, weather_penalty, bottom=bottom, label="Weather Penalty")
#         bottom = bottom + weather_penalty
#         ax.bar(x, delay_penalty, bottom=bottom, label="Delay Penalty")
#         bottom = bottom + delay_penalty
#         ax.bar(x, fuel_penalty, bottom=bottom, label="Fuel Penalty (x/1000)")
#         bottom = bottom + fuel_penalty
#         ax.bar(x, toll_penalty, bottom=bottom, label="Toll Delay (hr)")
#         ax.set_xticks(x)
#         ax.set_xticklabels(top5["Trip_ID"].astype(str), rotation=45)
#         ax.set_ylabel("Composite components (hours)")
#         ax.legend()
#         st.pyplot(fig)
#
#     # Model validation scatter
#     st.subheader("Model: Predicted vs Actual (sample)")
#     if diag.get("X_test") is not None:
#         y_test = diag["y_test"]
#         y_pred = diag["y_pred"]
#         fig2, ax2 = plt.subplots(figsize=(6, 6))
#         ax2.scatter(y_test, y_pred, alpha=0.6)
#         ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
#         ax2.set_xlabel("Actual ETA (hr)")
#         ax2.set_ylabel("Predicted ETA (hr)")
#         ax2.set_title(f"Predicted vs Actual (R2={diag['r2']:.3f})")
#         st.pyplot(fig2)
#         st.write("Model MAE (hrs):", f"{diag['mae']:.3f}")
#     else:
#         st.info("Not enough data to show model validation (small dataset).")
#
#     # ETA prediction small form
#     st.sidebar.subheader("Predict ETA for a new trip")
#     with st.sidebar.form("eta_form"):
#         in_distance = st.number_input("Distance (km)", min_value=1.0, value=100.0)
#         in_speed = st.number_input("Estimated Speed (km/h)", min_value=10.0, value=70.0)
#         in_weather = st.selectbox("Weather", options=["Clear", "Cloudy", "Rain", "Fog", "Snow"])
#         in_traffic = st.selectbox("Traffic", options=["Low", "Medium", "High"])
#         in_delay = st.number_input("Known delay (min)", min_value=0, value=0)
#         submitted = st.form_submit_button("Predict ETA")
#         if submitted:
#             road_type = "Highway" if in_distance > 200 else "City"
#             speed_limit = 80 if road_type == "Highway" else 50
#             adjusted_speed = min(in_speed, speed_limit)
#             base_travel_hr = in_distance / (adjusted_speed + 1e-6)
#             wf_map = {"Clear": 1.0, "Cloudy": 1.05, "Rain": 1.15, "Fog": 1.25, "Snow": 1.4}
#             tf_map = {"Low": 1.0, "Medium": 1.1, "High": 1.25}
#             weather_factor_val = wf_map[in_weather]
#             traffic_factor_val = tf_map[in_traffic]
#             toll_cost_est = estimate_toll(in_distance)
#             X_new = pd.DataFrame([{
#                 "Distance_km": in_distance,
#                 "adjusted_speed": adjusted_speed,
#                 "weather_factor": weather_factor_val,
#                 "traffic_factor": traffic_factor_val,
#                 "delay_min": in_delay,
#                 "toll_cost": toll_cost_est
#             }])
#             try:
#                 pred_eta = model.predict(X_new)[0]
#             except Exception:
#                 pred_eta = base_travel_hr + (in_delay / 60.0) + base_travel_hr * (weather_factor_val - 1.0) + base_travel_hr * (traffic_factor_val - 1.0)
#             st.sidebar.success(f"Predicted ETA: {pred_eta:.2f} hours ({pred_eta * 60:.0f} minutes)")
#             st.sidebar.write(f"Base travel time (hr): {base_travel_hr:.2f}")
#             st.sidebar.write(f"Adjusted speed used: {adjusted_speed} km/h")
#             st.sidebar.write(f"Estimated toll cost: ₹{toll_cost_est}")
#
#     # Export Top Routes
#     if st.button("Export Top Routes to CSV"):
#         out_path = "top_routes_export.csv"
#         top_routes.to_csv(out_path, index=False)
#         st.success(f"Top routes saved to {out_path}")
#
#     # Phase 4: Re-routing check (button)
#     if st.sidebar.button("Check Re-routing Now"):
#         new_route, reason = get_rerouted_path(current_route_id, reroute_weather, reroute_traffic, reroute_delay)
#         st.markdown("### 🔄 Re-routing Recommendation")
#         if new_route != current_route_id and reason is not None:
#             st.error(f"⚠️ {reason}")
#             st.success(f"**Suggested new route:** {new_route}")
#         else:
#             st.info("✅ No re-routing recommended - current route remains optimal.")
#
#     # -----------------------
#     # 🔍 DEBUG PANEL — SHOW RULE APPLIED
#     # -----------------------
#     st.subheader("🔍 Re-routing Debug Panel")
#     new_route, reason = get_rerouted_path(current_route_id, reroute_weather, reroute_traffic, reroute_delay)
#     st.write(f"**Current Route:** {current_route_id}")
#     st.write(f"**Weather Now:** {reroute_weather}")
#     st.write(f"**Traffic Now:** {reroute_traffic}")
#     st.write(f"**Accumulated Delay:** {reroute_delay} min")
#
#     if reason:
#         st.success(
#             f"### ✔ Re-routing Activated!\n"
#             f"**New Route:** {new_route}\n\n"
#             f"**Reason:** {reason}"
#         )
#     else:
#         st.info("ℹ No re-routing applied. Conditions are normal.")
#
#     # Footer
#     st.markdown("---")
#     st.caption("Smart Truck Routing — Prototype. Next steps: integrate live APIs (Google Maps, OpenWeather), add XGBoost model for ETA, and deploy.")
#
#
# if __name__ == "__main__":
#     main()

# dashboard.py
"""
Smart Truck Routing — Phase 3 & Phase 4 (ETA, Route Scoring, Re-routing)
Save as dashboard.py and run:
    streamlit run dashboard.py

This file keeps your earlier dashboard layout and features and **adds a working
Phase-4 re-routing system** with:
 - rule-based reroute decisions
 - a Re-route Test Panel (shows which rule fired)
 - debug text with the exact reason
 - safe fallbacks if model/data are missing
 - minimal changes to your original layout/visuals
"""

# import os
# from typing import Tuple, Optional
#
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score
#
# # -------------------------
# # Page config & visuals
# # -------------------------
# sns.set(style="whitegrid")
# st.set_page_config(page_title="Smart Truck Routing", layout="wide", page_icon="🚛")
#
# # -------------------------
# # Constants
# # -------------------------
# DATA_PATH = "clean_trips.csv"  # change if your file has a different name
# FUEL_PRICE = 95.0  # ₹/L used as default for fuel cost calculations
#
# # -------------------------
# # Utilities
# # -------------------------
#
#
# @st.cache_data
# def load_data(path: str = DATA_PATH) -> pd.DataFrame:
#     """Load CSV and normalize column names. Returns empty DataFrame on missing file."""
#     if not os.path.exists(path):
#         st.error(
#             f"Data file not found: {path}. Place your cleaned CSV in the app folder or edit DATA_PATH."
#         )
#         return pd.DataFrame()
#     df = pd.read_csv(path)
#     # normalize column names
#     df.columns = [c.strip() for c in df.columns]
#     return df
#
#
# def estimate_toll(distance_km: float) -> float:
#     """Estimate toll cost with simple buckets."""
#     if distance_km < 100:
#         return 100.0
#     elif distance_km < 300:
#         return 250.0
#     else:
#         return 500.0
#
#
# def ensure_columns_for_model(df: pd.DataFrame) -> pd.DataFrame:
#     """Ensure the dataframe has derived columns required by model & scoring.
#
#     This modifies df in-place (returns df).
#     """
#     # safe guards if Distance_km missing
#     if "Distance_km" not in df.columns:
#         df["Distance_km"] = 0.0
#
#     # travel_time_hr: use actual_duration_min or fallback estimate
#     if "travel_time_hr" not in df.columns:
#         if "actual_duration_min" in df.columns:
#             df["travel_time_hr"] = df["actual_duration_min"] / 60.0
#         else:
#             # fallback: distance / 50 km/h
#             df["travel_time_hr"] = df.get("Distance_km", 0.0) / 50.0
#
#     # computed_speed: Distance / actual_duration
#     if "computed_speed" not in df.columns:
#         if "actual_duration_min" in df.columns and (df["actual_duration_min"] > 0).any():
#             df["computed_speed"] = df["Distance_km"] / (df["actual_duration_min"] / 60.0 + 1e-6)
#         else:
#             df["computed_speed"] = df["Distance_km"] / (df["travel_time_hr"] + 1e-6)
#
#     df["computed_speed"] = df["computed_speed"].replace([np.inf, -np.inf], np.nan).fillna(50.0)
#
#     # unify delay column
#     if "delay_min" not in df.columns:
#         if "Delay_Minutes" in df.columns:
#             df["delay_min"] = df["Delay_Minutes"]
#         else:
#             df["delay_min"] = 0.0
#     df["delay_min"] = df["delay_min"].fillna(df["delay_min"].median())
#
#     # weather factor mapping
#     wf_map = {"Clear": 1.0, "Cloudy": 1.05, "Rain": 1.15, "Fog": 1.25, "Snow": 1.4, "Storm": 1.5}
#     if "Weather_Condition" in df.columns:
#         df["weather_factor"] = df["Weather_Condition"].map(wf_map).fillna(1.1)
#     else:
#         df["weather_factor"] = 1.1
#
#     # traffic factor mapping (unify possible column names)
#     traffic_map = {"Low": 1.0, "Light": 1.0, "Medium": 1.1, "Heavy": 1.25, "High": 1.25, "Jam": 1.4}
#     if "Traffic_Condition" in df.columns:
#         df["traffic_factor"] = df["Traffic_Condition"].map(traffic_map).fillna(1.1)
#     elif "Traffic_Level" in df.columns:
#         df["traffic_factor"] = df["Traffic_Level"].map(traffic_map).fillna(1.1)
#     elif "Traffic" in df.columns:
#         df["traffic_factor"] = df["Traffic"].map(traffic_map).fillna(1.1)
#     else:
#         df["traffic_factor"] = 1.1
#
#     # fuel efficiency
#     if "fuel_efficiency_kmpl" not in df.columns:
#         if "Fuel_Consumption_L" in df.columns and "Distance_km" in df.columns:
#             df["fuel_efficiency_kmpl"] = df["Distance_km"] / (df["Fuel_Consumption_L"] + 1e-6)
#         else:
#             df["fuel_efficiency_kmpl"] = 3.7
#
#     # speed limits and adjusted speed
#     if "speed_limit" not in df.columns:
#         df["road_type"] = np.where(df["Distance_km"] > 200, "Highway", "City")
#         speed_limits = {"Highway": 80.0, "City": 50.0, "Rural": 60.0}
#         df["speed_limit"] = df["road_type"].map(speed_limits).fillna(60.0)
#
#     df["adjusted_speed"] = np.minimum(df["computed_speed"], df["speed_limit"]).fillna(50.0)
#
#     # recompute travel_time_hr safely
#     df["travel_time_hr"] = df["Distance_km"] / (df["adjusted_speed"] + 1e-6)
#
#     # toll cost / toll delay proxies
#     if "toll_cost" not in df.columns:
#         df["toll_cost"] = df["Distance_km"].apply(estimate_toll)
#
#     if "toll_delay" not in df.columns:
#         df["toll_delay"] = df["Distance_km"].apply(lambda d: 0.1 if d < 100 else (0.2 if d < 300 else 0.3))
#
#     # ensure Trip_ID exists to display
#     if "Trip_ID" not in df.columns:
#         df["Trip_ID"] = np.arange(len(df)).astype(int)
#
#     return df
#
#
# # -------------------------
# # Baseline ETA model
# # -------------------------
# @st.cache_resource
# def train_eta_model(df: pd.DataFrame) -> Tuple[LinearRegression, dict]:
#     """Train a simple linear regression ETA model with safe fallbacks.
#
#     Returns (model, diagnostics). diagnostics contains mae, r2, features, X_test, y_test, y_pred
#     """
#     d = df.copy()
#     features = ["Distance_km", "adjusted_speed", "weather_factor", "traffic_factor", "delay_min", "toll_cost"]
#
#     # ensure features exist
#     for f in features:
#         if f not in d.columns:
#             if f == "adjusted_speed":
#                 d[f] = np.minimum(d.get("computed_speed", 50.0), d.get("speed_limit", 60.0)).fillna(50.0)
#             else:
#                 d[f] = d.get(f, 0.0)
#
#     # define target
#     if "ETA_hr" not in d.columns:
#         if "actual_duration_min" in d.columns:
#             d["ETA_hr"] = d["actual_duration_min"] / 60.0
#         else:
#             # construct synthetic ETA target as fallback
#             d["ETA_hr"] = (
#                 d["travel_time_hr"]
#                 + d["travel_time_hr"] * (d["weather_factor"] - 1.0)
#                 + d["travel_time_hr"] * (d["traffic_factor"] - 1.0)
#                 + d["delay_min"] / 60.0
#                 + d["toll_delay"]
#             )
#
#     # drop rows missing features/target
#     d = d.dropna(subset=features + ["ETA_hr"], how="any")
#     if d.shape[0] < 30:
#         # Not enough data: return dummy model + empty diagnostics
#         dummy = LinearRegression()
#         diagnostics = {"mae": None, "r2": None, "features": features, "X_test": None, "y_test": None, "y_pred": None}
#         return dummy, diagnostics
#
#     X = d[features]
#     y = d["ETA_hr"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     diagnostics = {"mae": mae, "r2": r2, "features": features, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}
#     return model, diagnostics
#
#
# # -------------------------
# # Phase 4: Re-routing logic (rule-based)
# # -------------------------
# def get_rerouted_path(current_route: str, weather: Optional[str], traffic_level: Optional[str], delay_min: float) -> Tuple[str, Optional[str]]:
#     """
#     Simple rule-based rerouting (deterministic and easy to extend).
#
#     Returns (new_route_id, reason) where reason is None if no reroute applied.
#     Priority: severe weather > high delay > heavy traffic.
#     """
#     # guard and normalize inputs
#     w = (str(weather) if weather is not None else "").title()
#     t = (str(traffic_level) if traffic_level is not None else "").title()
#     new_route = current_route
#     reason = None
#
#     # Rule: severe weather
#     if w in ["Snow", "Storm"]:
#         new_route = f"{current_route}_ALT_WEATHER"
#         reason = f"Severe weather detected: {w}"
#         return new_route, reason
#
#     # Rule: very low visibility (Fog) -> prefer highways / weather-safe route
#     if w == "Fog":
#         new_route = f"{current_route}_ALT_FOG"
#         reason = "Low visibility (Fog) — use fog-safe route"
#         return new_route, reason
#
#     # Rule: accumulated delay too high
#     if delay_min > 90:
#         new_route = f"{current_route}_ALT_DELAY"
#         reason = f"High accumulated delay: {delay_min:.0f} minutes"
#         return new_route, reason
#
#     # Rule: heavy traffic
#     if t in ["High", "Heavy", "Jam"]:
#         new_route = f"{current_route}_ALT_TRAFFIC"
#         reason = f"Heavy traffic detected: {t}"
#         return new_route, reason
#
#     # Rule: medium traffic + rain -> medium-safe route
#     if t == "Medium" and w in ["Rain", "Cloudy"]:
#         new_route = f"{current_route}_ALT_MED"
#         reason = f"Medium traffic with {w} -> choose medium-safe alternative"
#         return new_route, reason
#
#     # No rule matched -> keep current route
#     return new_route, None
#
#
# # -------------------------
# # Main app
# # -------------------------
# def main():
#     st.title("🚚 Smart Truck Routing — ETA, Scoring & Re-routing")
#     st.markdown(
#         "Baseline ETA model + Route ranking & visualizations. "
#         "Use the left sidebar to filter data and test re-routing (Phase 4)."
#     )
#
#     # Load data
#     df = load_data(DATA_PATH)
#     if df.empty:
#         return
#
#     # Prepare derived columns
#     df = ensure_columns_for_model(df)
#
#     # Sidebar: Filters & re-route test inputs
#     st.sidebar.header("Filters & Re-route Input")
#     start_options = ["All"] + sorted(df["Start_Location"].dropna().unique().tolist()) if "Start_Location" in df.columns else ["All"]
#     end_options = ["All"] + sorted(df["End_Location"].dropna().unique().tolist()) if "End_Location" in df.columns else ["All"]
#     start_loc = st.sidebar.selectbox("Start Location", options=start_options)
#     end_loc = st.sidebar.selectbox("End Location", options=end_options)
#
#     show_live_weather = st.sidebar.checkbox("Show Live Weather for selected start", value=False)
#     show_live_traffic = st.sidebar.checkbox("Show Live Traffic check (Google Directions)", value=False)
#
#     st.sidebar.markdown("### Re-route Test Inputs")
#     current_route_id = st.sidebar.text_input("Current Route ID", value="R1000")
#     origin_input = st.sidebar.text_input("Origin (address or Start_Location)", value="")
#     dest_input = st.sidebar.text_input("Destination (address or End_Location)", value="")
#     reroute_manual_delay = st.sidebar.number_input("Accumulated delay (min) for test", min_value=0, value=0, step=5)
#
#     weather_filter = st.sidebar.multiselect("Weather (filter)", options=sorted(df.get("Weather_Condition", pd.Series()).dropna().unique().tolist()))
#     traffic_select = st.sidebar.selectbox("Traffic (display only)", options=["Low", "Medium", "High", "Jam"])
#
#     limit_rows = st.sidebar.slider("Plot sample size (for performance)", min_value=500, max_value=min(5000, len(df)), value=min(2000, len(df)), step=500)
#
#     # Apply filters
#     dff = df.copy()
#     if start_loc != "All":
#         dff = dff[dff["Start_Location"] == start_loc]
#     if end_loc != "All":
#         dff = dff[dff["End_Location"] == end_loc]
#     if weather_filter:
#         dff = dff[dff["Weather_Condition"].isin(weather_filter)]
#
#     st.markdown(f"**Filtered rows:** {dff.shape[0]}")
#
#     # Train baseline ETA model
#     with st.spinner("Training baseline ETA model..."):
#         model, diag = train_eta_model(df)
#
#     # Sidebar: model diagnostics
#     st.sidebar.subheader("Model diagnostics (baseline ETA)")
#     if diag.get("r2") is not None:
#         st.sidebar.write(f"R²: {diag['r2']:.3f}")
#         st.sidebar.write(f"MAE (hrs): {diag['mae']:.3f}")
#     else:
#         st.sidebar.write("Not enough data to compute diagnostics.")
#
#     # Compute composite route score for filtered df
#     if "travel_time_hr" not in dff.columns:
#         dff["travel_time_hr"] = dff["Distance_km"] / (dff["adjusted_speed"] + 1e-6)
#     if "fuel_efficiency_kmpl" not in dff.columns:
#         dff["fuel_efficiency_kmpl"] = dff.get("fuel_efficiency_kmpl", 3.7)
#
#     dff["fuel_cost"] = (dff["Distance_km"] / (dff["fuel_efficiency_kmpl"] + 1e-9)) * FUEL_PRICE
#     dff["weather_penalty"] = dff["travel_time_hr"] * (dff["weather_factor"] - 1.0)
#     dff["delay_penalty"] = (dff["delay_min"] / 60.0) * 0.5
#     if "toll_delay" not in dff.columns:
#         dff["toll_delay"] = dff["Distance_km"].apply(lambda d: 0.1 if d < 100 else (0.2 if d < 300 else 0.3))
#
#     dff["route_score"] = (
#         dff["travel_time_hr"]
#         + dff["weather_penalty"]
#         + dff["delay_penalty"]
#         + (dff["fuel_cost"] / 1000.0)
#         + dff["toll_delay"]
#     )
#     dff["route_rank"] = dff["route_score"].rank(method="dense")
#
#     # Top routes table
#     st.subheader("Top Recommended Routes (by composite score)")
#     top_routes = dff.sort_values("route_score").head(10).reset_index(drop=True)
#     display_cols = [c for c in ["Trip_ID", "Start_Location", "End_Location", "Distance_km", "travel_time_hr", "fuel_cost", "delay_min", "Weather_Condition", "route_score", "route_rank"] if c in top_routes.columns]
#     st.dataframe(top_routes[display_cols].head(10), use_container_width=True)
#
#     # KPIs
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Filtered Trips", dff.shape[0])
#     col2.metric("Avg Delay (min)", f"{dff['delay_min'].mean():.1f}")
#     col3.metric("Avg Fuel Efficiency (km/l)", f"{dff['fuel_efficiency_kmpl'].mean():.2f}")
#
#     # Route score breakdown (top 5)
#     st.subheader("Route Score Breakdown (Top 5)")
#     top5 = top_routes.head(5).copy()
#     if not top5.empty:
#         fig, ax = plt.subplots(figsize=(10, 5))
#         x = np.arange(len(top5))
#         travel_time = top5["travel_time_hr"].values
#         weather_penalty = top5["weather_penalty"].values
#         delay_penalty = top5["delay_penalty"].values
#         fuel_penalty = (top5["fuel_cost"] / 1000.0).values
#         toll_penalty = top5["toll_delay"].values
#         bottom = np.zeros_like(travel_time)
#         ax.bar(x, travel_time, label="Travel Time (hr)")
#         bottom = bottom + travel_time
#         ax.bar(x, weather_penalty, bottom=bottom, label="Weather Penalty")
#         bottom = bottom + weather_penalty
#         ax.bar(x, delay_penalty, bottom=bottom, label="Delay Penalty")
#         bottom = bottom + delay_penalty
#         ax.bar(x, fuel_penalty, bottom=bottom, label="Fuel Penalty (x/1000)")
#         bottom = bottom + fuel_penalty
#         ax.bar(x, toll_penalty, bottom=bottom, label="Toll Delay (hr)")
#         ax.set_xticks(x)
#         ax.set_xticklabels(top5["Trip_ID"].astype(str), rotation=45)
#         ax.set_ylabel("Composite components (hours)")
#         ax.legend()
#         st.pyplot(fig)
#
#     # Model validation plot
#     st.subheader("Model: Predicted vs Actual (sample)")
#     if diag.get("X_test") is not None:
#         try:
#             y_test = diag["y_test"]
#             y_pred = diag["y_pred"]
#             fig2, ax2 = plt.subplots(figsize=(6, 6))
#             ax2.scatter(y_test, y_pred, alpha=0.6)
#             ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
#             ax2.set_xlabel("Actual ETA (hr)")
#             ax2.set_ylabel("Predicted ETA (hr)")
#             ax2.set_title(f"Predicted vs Actual (R2={diag['r2']:.3f})")
#             st.pyplot(fig2)
#             st.write("Model MAE (hrs):", f"{diag['mae']:.3f}")
#         except Exception:
#             st.info("Could not render model validation plot.")
#     else:
#         st.info("Not enough data to show model validation (small dataset).")
#
#     # ETA prediction small form in sidebar
#     st.sidebar.subheader("Predict ETA for a new trip")
#     with st.sidebar.form("eta_form"):
#         in_distance = st.number_input("Distance (km)", min_value=1.0, value=100.0)
#         in_speed = st.number_input("Estimated Speed (km/h)", min_value=10.0, value=70.0)
#         in_weather = st.selectbox("Weather", options=["Clear", "Cloudy", "Rain", "Fog", "Snow"])
#         in_traffic = st.selectbox("Traffic", options=["Low", "Medium", "High"])
#         in_delay = st.number_input("Known delay (min)", min_value=0, value=0)
#         submitted = st.form_submit_button("Predict ETA")
#         if submitted:
#             road_type = "Highway" if in_distance > 200 else "City"
#             speed_limit = 80 if road_type == "Highway" else 50
#             adjusted_speed = min(in_speed, speed_limit)
#             base_travel_hr = in_distance / (adjusted_speed + 1e-6)
#             wf_map = {"Clear": 1.0, "Cloudy": 1.05, "Rain": 1.15, "Fog": 1.25, "Snow": 1.4}
#             tf_map = {"Low": 1.0, "Medium": 1.1, "High": 1.25}
#             weather_factor_val = wf_map[in_weather]
#             traffic_factor_val = tf_map[in_traffic]
#             toll_cost_est = estimate_toll(in_distance)
#             X_new = pd.DataFrame([{
#                 "Distance_km": in_distance,
#                 "adjusted_speed": adjusted_speed,
#                 "weather_factor": weather_factor_val,
#                 "traffic_factor": traffic_factor_val,
#                 "delay_min": in_delay,
#                 "toll_cost": toll_cost_est
#             }])
#             try:
#                 pred_eta = model.predict(X_new)[0]
#             except Exception:
#                 # fallback heuristic
#                 pred_eta = base_travel_hr + (in_delay / 60.0) + base_travel_hr * (weather_factor_val - 1.0) + base_travel_hr * (traffic_factor_val - 1.0)
#             st.sidebar.success(f"Predicted ETA: {pred_eta:.2f} hours ({pred_eta * 60:.0f} minutes)")
#             st.sidebar.write(f"Base travel time (hr): {base_travel_hr:.2f}")
#             st.sidebar.write(f"Adjusted speed used: {adjusted_speed} km/h")
#             st.sidebar.write(f"Estimated toll cost: ₹{toll_cost_est}")
#
#     # Export top routes
#     if st.button("Export Top Routes to CSV"):
#         out_path = "top_routes_export.csv"
#         top_routes.to_csv(out_path, index=False)
#         st.success(f"Top routes saved to {out_path}")
#
#     # -------------------------
#     # Phase 4: Combined Re-routing logic demo (button)
#     # -------------------------
#     st.markdown("## 🔄 Re-routing Decision (combined rule-based)")
#     # The UI uses the re-route test inputs from sidebar
#     # Determine "current" weather & traffic for this test:
#     # prefer live options (if you wire APIs later), else use manual controls in sidebar
#     test_weather = None
#     # try pick first weather from filter or fallback to top_routes' Weather_Condition
#     if len(weather_filter) > 0:
#         test_weather = weather_filter[0]
#     elif "Weather_Condition" in top_routes.columns and top_routes.shape[0] > 0:
#         test_weather = top_routes.iloc[0].get("Weather_Condition", None)
#     else:
#         test_weather = "Clear"
#
#     test_traffic = traffic_select  # manual sidebar selector
#     test_delay = float(reroute_manual_delay)
#
#     # Button to check reroute now:
#     if st.button("Evaluate Re-routing (using test inputs)"):
#         new_route, reason = get_rerouted_path(current_route_id, test_weather, test_traffic, test_delay)
#         st.markdown("### 🔁 Re-routing Recommendation")
#         if reason:
#             st.error(f"⚠️ {reason}")
#             st.success(f"Suggested new route: **{new_route}**")
#         else:
#             st.info("✅ No re-routing recommended - current route remains optimal.")
#
#     # -------------------------
#     # 🔍 Re-routing Debug/Test Panel (always visible)
#     # -------------------------
#     st.markdown("## 🔍 Re-routing Debug Panel (Test Inputs)")
#     st.write(f"**Current Route (sidebar):** {current_route_id}")
#     st.write(f"**Origin input:** {origin_input or '(none)'}")
#     st.write(f"**Destination input:** {dest_input or '(none)'}")
#     st.write(f"**Weather for test:** {test_weather}")
#     st.write(f"**Traffic for test:** {test_traffic}")
#     st.write(f"**Accumulated delay (min):** {test_delay}")
#
#     # Always compute & show rule result in debug panel
#     debug_new_route, debug_reason = get_rerouted_path(current_route_id, test_weather, test_traffic, test_delay)
#     if debug_reason:
#         st.success(f"✔ Re-routing activated: {debug_new_route}")
#         st.write(f"**Reason:** {debug_reason}")
#     else:
#         st.info("ℹ️ No re-routing applied with the current test inputs.")
#
#     # Footer
#     st.markdown("---")
#     st.caption("Smart Truck Routing — Prototype. Next steps: integrate live APIs (Google Maps, OpenWeather), add stronger ML model (XGBoost), and deploy.")
#
#
# if __name__ == "__main__":
#     main()R
#
# dashboard.py
"""Smart Truck Routing — Phase 3/4/5 (ETA, Route Scoring, Re-routing, Live APIs, Combined Decision)
Save as dashboard.py and run:
    streamlit run dashboard.py
"""
import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Optional imports
try:
    import requests
except Exception:
    requests = None

try:
    import joblib
except Exception:
    joblib = None

import requests


def fetch_weather(city, api_key):
    """
    Safely fetch weather data from OpenWeatherMap.
    Returns: dict containing 'main', 'description', 'temp', 'visibility', 'wind'.
    If API fails → returns None (so the dashboard never breaks).
    """
    if not api_key or not city:
        return None

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get("cod") != 200:
            return None

        return {
            "main": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "temp": data["main"]["temp"],
            "visibility": data.get("visibility"),
            "wind": data["wind"]["speed"]
        }

    except:
        return None


# -------------------------#
# Page config & visuals
# -------------------------#
sns.set(style="whitegrid")
st.set_page_config(page_title="Vehicle Route Optimizer", layout="wide", page_icon="🚛")

# -------------------------#
# Constants
# -------------------------#
DATA_PATH = "clean_trips.csv"  # change if your file has a different name
FUEL_PRICE = 95.0  # ₹/L used as default for fuel cost calculations

# -------------------------#
# Utilities
# -------------------------#
@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV and normalize column names. Returns empty DataFrame on missing file."""
    if not os.path.exists(path):
        st.error(
            f"Data file not found: {path}. Place your cleaned CSV in the app folder or edit DATA_PATH."
        )
        return pd.DataFrame()
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def estimate_toll(distance_km: float) -> float:
    """Estimate toll cost with simple buckets."""
    if distance_km < 100:
        return 100.0
    elif distance_km < 300:
        return 250.0
    else:
        return 500.0


def ensure_columns_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has derived columns required by model & scoring.
    This modifies df in-place (returns df).
    """
    # safe guards if Distance_km missing
    if "Distance_km" not in df.columns:
        df["Distance_km"] = 0.0
    # travel_time_hr: use actual_duration_min or fallback estimate
    if "travel_time_hr" not in df.columns:
        if "actual_duration_min" in df.columns:
            df["travel_time_hr"] = df["actual_duration_min"] / 60.0
        else:
            # fallback: distance / 50 km/h
            df["travel_time_hr"] = df.get("Distance_km", 0.0) / 50.0
    # computed_speed: Distance / actual_duration
    if "computed_speed" not in df.columns:
        if "actual_duration_min" in df.columns and (df["actual_duration_min"] > 0).any():
            df["computed_speed"] = df["Distance_km"] / (df["actual_duration_min"] / 60.0 + 1e-6)
        else:
            df["computed_speed"] = df["Distance_km"] / (df["travel_time_hr"] + 1e-6)
    df["computed_speed"] = df["computed_speed"].replace([np.inf, -np.inf], np.nan).fillna(50.0)
    # unify delay column
    if "delay_min" not in df.columns:
        if "Delay_Minutes" in df.columns:
            df["delay_min"] = df["Delay_Minutes"]
        else:
            df["delay_min"] = 0.0
    df["delay_min"] = df["delay_min"].fillna(df["delay_min"].median())
    # weather factor mapping
    wf_map = {"Clear": 1.0, "Cloudy": 1.05, "Rain": 1.15, "Fog": 1.25, "Snow": 1.4, "Storm": 1.5}
    if "Weather_Condition" in df.columns:
        df["weather_factor"] = df["Weather_Condition"].map(wf_map).fillna(1.1)
    else:
        df["weather_factor"] = 1.1
    # traffic factor mapping (unify possible column names)
    traffic_map = {"Low": 1.0, "Light": 1.0, "Medium": 1.1, "Heavy": 1.25, "High": 1.25, "Jam": 1.4}
    if "Traffic_Condition" in df.columns:
        df["traffic_factor"] = df["Traffic_Condition"].map(traffic_map).fillna(1.1)
    elif "Traffic_Level" in df.columns:
        df["traffic_factor"] = df["Traffic_Level"].map(traffic_map).fillna(1.1)
    elif "Traffic" in df.columns:
        df["traffic_factor"] = df["Traffic"].map(traffic_map).fillna(1.1)
    else:
        df["traffic_factor"] = 1.1
    # fuel efficiency
    if "fuel_efficiency_kmpl" not in df.columns:
        if "Fuel_Consumption_L" in df.columns and "Distance_km" in df.columns:
            df["fuel_efficiency_kmpl"] = df["Distance_km"] / (df["Fuel_Consumption_L"] + 1e-6)
        else:
            df["fuel_efficiency_kmpl"] = 3.7
    # speed limits and adjusted speed
    if "speed_limit" not in df.columns:
        df["road_type"] = np.where(df["Distance_km"] > 200, "Highway", "City")
        speed_limits = {"Highway": 80.0, "City": 50.0, "Rural": 60.0}
        df["speed_limit"] = df["road_type"].map(speed_limits).fillna(60.0)
    df["adjusted_speed"] = np.minimum(df["computed_speed"], df["speed_limit"]).fillna(50.0)
    # recompute travel_time_hr safely
    df["travel_time_hr"] = df["Distance_km"] / (df["adjusted_speed"] + 1e-6)
    # toll cost / toll delay proxies
    if "toll_cost" not in df.columns:
        df["toll_cost"] = df["Distance_km"].apply(estimate_toll)
    if "toll_delay" not in df.columns:
        df["toll_delay"] = df["Distance_km"].apply(lambda d: 0.1 if d < 100 else (0.2 if d < 300 else 0.3))
    # ensure Trip_ID exists to display
    if "Trip_ID" not in df.columns:
        df["Trip_ID"] = np.arange(len(df)).astype(int)
    return df


# -------------------------#
# Baseline ETA model
# -------------------------#
@st.cache_resource
def train_eta_model(df: pd.DataFrame) -> Tuple[LinearRegression, dict]:
    """Train a simple linear regression ETA model with safe fallbacks.
    Returns (model, diagnostics). diagnostics contains mae, r2, features, X_test, y_test, y_pred
    """
    d = df.copy()
    features = ["Distance_km", "adjusted_speed", "weather_factor", "traffic_factor", "delay_min", "toll_cost"]
    # ensure features exist
    for f in features:
        if f not in d.columns:
            if f == "adjusted_speed":
                d[f] = np.minimum(d.get("computed_speed", 50.0), d.get("speed_limit", 60.0)).fillna(50.0)
            else:
                d[f] = d.get(f, 0.0)
    # define target
    if "ETA_hr" not in d.columns:
        if "actual_duration_min" in d.columns:
            d["ETA_hr"] = d["actual_duration_min"] / 60.0
        else:
            # construct synthetic ETA target as fallback
            d["ETA_hr"] = (
                d["travel_time_hr"]
                + d["travel_time_hr"] * (d["weather_factor"] - 1.0)
                + d["travel_time_hr"] * (d["traffic_factor"] - 1.0)
                + d["delay_min"] / 60.0
                + d["toll_delay"]
            )
    # drop rows missing features/target
    d = d.dropna(subset=features + ["ETA_hr"], how="any")
    if d.shape[0] < 30:
        # Not enough data: return dummy model + empty diagnostics
        dummy = LinearRegression()
        diagnostics = {"mae": None, "r2": None, "features": features, "X_test": None, "y_test": None, "y_pred": None}
        return dummy, diagnostics
    X = d[features]
    y = d["ETA_hr"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    diagnostics = {"mae": mae, "r2": r2, "features": features, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}
    return model, diagnostics


# -------------------------#
# Phase 4: Re-routing logic (rule-based)
# -------------------------#
def get_rerouted_path(current_route: str, weather: Optional[str], traffic_level: Optional[str], delay_min: float) -> Tuple[str, Optional[str]]:
    """
    Simple rule-based rerouting (deterministic and easy to extend).
    Returns (new_route_id, reason) where reason is None if no reroute applied.
    Priority: severe weather > high delay > heavy traffic.
    """
    # guard and normalize inputs
    w = (str(weather) if weather is not None else "").title()
    t = (str(traffic_level) if traffic_level is not None else "").title()
    new_route = current_route
    reason = None
    # Rule: severe weather
    if w in ["Snow", "Storm"]:
        new_route = f"{current_route}_ALT_WEATHER"
        reason = f"Severe weather detected: {w}"
        return new_route, reason
    # Rule: very low visibility (Fog) -> prefer highways / weather-safe route
    if w == "Fog":
        new_route = f"{current_route}_ALT_FOG"
        reason = "Low visibility (Fog) — use fog-safe route"
        return new_route, reason
    # Rule: accumulated delay too high
    if delay_min > 90:
        new_route = f"{current_route}_ALT_DELAY"
        reason = f"High accumulated delay: {delay_min:.0f} minutes"
        return new_route, reason
    # Rule: heavy traffic
    if t in ["High", "Heavy", "Jam"]:
        new_route = f"{current_route}_ALT_TRAFFIC"
        reason = f"Heavy traffic detected: {t}"
        return new_route, reason
    # Rule: medium traffic + rain -> medium-safe route
    if t == "Medium" and w in ["Rain", "Cloudy"]:
        new_route = f"{current_route}_ALT_MED"
        reason = f"Medium traffic with {w} -> choose medium-safe alternative"
        return new_route, reason
    # No rule matched -> keep current route
    return new_route, None


# -------------------------
# Phase 4/5: Live APIs helpers (optional)
# -------------------------
def geocode_address_google(address: str, api_key: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) for address using Google Geocoding API. Returns None on failure."""
    if requests is None:
        return None
    if not api_key or not address:
        return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    try:
        r = requests.get(url, params=params, timeout=6)
        j = r.json()
        if j.get("status") == "OK" and j.get("results"):
            loc = j["results"][0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception:
        return None
    return None


def get_live_weather_openweather(lat: float, lon: float, api_key: str) -> Optional[dict]:
    """Fetch current weather using OpenWeather OneCall or current weather API. Return dict or None."""
    if requests is None:
        return None
    if not api_key:
        return None
    # use current weather data endpoint
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def parse_openweather_to_condition(j: dict) -> Optional[str]:
    """Map OpenWeather JSON to our textual condition (Clear, Rain, Fog, Snow, Cloudy, Storm...)"""
    if not j:
        return None
    weather_lst = j.get("weather", [])
    if len(weather_lst) == 0:
        return None
    main = weather_lst[0].get("main", "").lower()
    if "rain" in main:
        return "Rain"
    if "cloud" in main:
        return "Cloudy"
    if "snow" in main:
        return "Snow"
    if "fog" in main or "mist" in main or "haze" in main:
        return "Fog"
    if "storm" in main or "thunder" in main:
        return "Storm"
    if "clear" in main:
        return "Clear"
    return None


def google_directions_traffic(origin: str, destination: str, api_key: str) -> Optional[dict]:
    """Call Google Directions API (traffic-aware) and return parsed summary:
       {'distance_km': float, 'duration_min': float, 'duration_in_traffic_min': float, 'polyline': str, 'raw': ...}
       On failure returns None.
    """
    if requests is None or not api_key or not origin or not destination:
        return None
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "key": api_key,
        "departure_time": "now",  # traffic-aware
        "alternatives": "true",
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        j = r.json()
        if j.get("status") != "OK" or not j.get("routes"):
            return None
        # choose best route by duration_in_traffic if present else duration
        best = None
        best_val = float("inf")
        for route in j["routes"]:
            legs = route.get("legs", [])
            if not legs:
                continue
            leg = legs[0]
            # prefer duration_in_traffic if present
            dur_traffic = None
            if "duration_in_traffic" in leg:
                dur_traffic = leg["duration_in_traffic"]["value"] / 60.0  # minutes->min
            dur = leg["duration"]["value"] / 60.0
            distance_km = leg["distance"]["value"] / 1000.0
            score = dur_traffic if dur_traffic is not None else dur
            if score < best_val:
                best_val = score
                best = {"duration_min": dur, "duration_in_traffic_min": dur_traffic, "distance_km": distance_km, "polyline": route.get("overview_polyline", {}).get("points"), "raw": route}
        return best
    except Exception:
        return None


# -------------------------
# Phase 5: Combined decision engine
# -------------------------
def combined_reroute_decision(
    current_route: str,
    rule_new_route: str,
    rule_reason: Optional[str],
    google_info: Optional[dict],
    ml_eta_hr: Optional[float],
    manual_delay_min: float,
) -> Tuple[str, str]:
    """
    Compare rule-based suggestion, google traffic-aware info and ML ETA to pick a final action.
    Returns (final_route_id, explanation_text). The function is conservative: if google or ML disagree,
    we combine reasons; if APIs fail, we fall back to rule-based result.
    """
    # If rule forced a weather/delay decision -> keep it as high priority
    if rule_reason:
        explanation = f"Rule-based re-route applied: {rule_reason}"
        return rule_new_route, explanation

    # If google info exists, use its duration_in_traffic (or duration)
    google_minutes = None
    if google_info:
        google_minutes = google_info.get("duration_in_traffic_min") or google_info.get("duration_min")

    # If we have ML predicted ETA (in hours), convert to minutes
    ml_minutes = None
    if ml_eta_hr is not None:
        ml_minutes = ml_eta_hr * 60.0

    # Heuristic:
    # - If google indicates traffic increase > 20% relative to baseline (duration), recommend alt
    # - If ML predicts significantly lower ETA for an alternative (this requires alternative features; we keep simple)
    # Here we do a simple check: if google_minutes exists and > 60 min and manual_delay > 45 -> reroute
    if google_minutes is not None:
        if google_minutes > 60 and manual_delay_min > 45:
            return f"{current_route}_ALT_GOOGLE", f"Google Directions shows high travel time ({google_minutes:.0f} min) + manual delay {manual_delay_min:.0f} min -> suggest traffic diversion"
        # if google shows moderate traffic but ML is much better, prefer ML suggestion
        if ml_minutes is not None and ml_minutes + 15 < google_minutes:
            return f"{current_route}_ALT_ML", f"ML ETA (~{ml_minutes:.0f} min) is better than Google traffic (~{google_minutes:.0f} min); choose ML-suggested route"
        # else keep original
        return current_route, "Google traffic okay — keep current route"

    # if no google info but ML suggests much worse ETA (over threshold), use ML
    if ml_minutes is not None:
        if ml_minutes > 180 and manual_delay_min > 60:
            return f"{current_route}_ALT_ML", f"ML predicts long ETA (~{ml_minutes:.0f} min) and manual delay high -> consider alternate ML route"
        return current_route, "ML did not warrant rerouting"

    # fallback: no APIs -> no change
    return current_route, "No live API data — fallback to rule-based (no change)"


# -------------------------
# Main app
# -------------------------
def main():
    st.title("🚚 Smart Truck Routing — ETA, Scoring & Re-routing")
    st.markdown(
        "Baseline ETA model + Route ranking & visualizations. "
        "Use the left sidebar to filter data and test re-routing (Phase 4/5)."
    )

    # Load data
    df = load_data(DATA_PATH)
    if df.empty:
        return

    # Prepare derived columns
    df = ensure_columns_for_model(df)

    # Sidebar: Filters & re-route test inputs
    st.sidebar.header("Filters & Re-route Input")
    start_options = ["All"] + sorted(df["Start_Location"].dropna().unique().tolist()) if "Start_Location" in df.columns else ["All"]
    end_options = ["All"] + sorted(df["End_Location"].dropna().unique().tolist()) if "End_Location" in df.columns else ["All"]
    start_loc = st.sidebar.selectbox("Start Location", options=start_options)
    end_loc = st.sidebar.selectbox("End Location", options=end_options)

    weather_api_key = st.sidebar.text_input("OpenWeather API Key")
    weather_city = st.sidebar.text_input("City for Live Weather")


    # 🌤 LIVE WEATHER PANEL
    # ---------------------

    st.sidebar.markdown("### 🌤 Live Weather")

    # Only run API if both fields are filled
    if weather_api_key and weather_city:
        wx = fetch_weather(weather_city, weather_api_key)

        if wx:
            st.sidebar.success(
                f"**{wx['main']}** — {wx['description']}**\n"
                f"🌡 Temperature: {wx.get('temp', 'N/A')}°C\n"
                f"💨 Wind: {wx.get('wind', 'N/A')} m/s\n"
                f"👁 Visibility: {wx.get('visibility', 'N/A')} m"
            )
        else:
            st.sidebar.warning("⚠ Could not fetch weather. Check city or API key.")
    else:
        st.sidebar.info("Enter API key and city to view live weather.")

    # Live API toggles & keys (optional)
    st.sidebar.markdown("### Live API integration (optional)")
    use_live_weather = st.sidebar.checkbox("Use OpenWeather live weather", value=False)
    use_live_google = st.sidebar.checkbox("Use Google Directions (traffic-aware)", value=False)
    # allow user to paste keys in sidebar for quick testing
    openweather_key_input = st.sidebar.text_input("OpenWeather API key (or set OPENWEATHER_API_KEY env var)", value=os.getenv("OPENWEATHER_API_KEY", ""))
    google_maps_key_input = st.sidebar.text_input("Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)", value=os.getenv("GOOGLE_MAPS_API_KEY", ""))

    # Re-route test inputs
    st.sidebar.markdown("### Re-route Test Inputs")
    current_route_id = st.sidebar.text_input("Current Route ID", value="R1000")
    origin_input = st.sidebar.text_input("Origin (address or Start_Location)", value="")
    dest_input = st.sidebar.text_input("Destination (address or End_Location)", value="")
    reroute_manual_delay = st.sidebar.number_input("Accumulated delay (min) for test", min_value=0, value=0, step=5)
    weather_filter = st.sidebar.multiselect("Weather (filter)", options=sorted(df.get("Weather_Condition", pd.Series()).dropna().unique().tolist()))
    traffic_select = st.sidebar.selectbox("Traffic (display only)", options=["Low", "Medium", "High", "Jam"])
    limit_rows = st.sidebar.slider("Plot sample size (for performance)", min_value=500, max_value=min(5000, len(df)), value=min(2000, len(df)), step=500)

    # Apply filters
    dff = df.copy()
    if start_loc != "All":
        dff = dff[dff["Start_Location"] == start_loc]
    if end_loc != "All":
        dff = dff[dff["End_Location"] == end_loc]
    if weather_filter:
        dff = dff[dff["Weather_Condition"].isin(weather_filter)]
    st.markdown(f"**Filtered rows:** {dff.shape[0]}")

    # Train baseline ETA model
    with st.spinner("Training baseline ETA model..."):
        model, diag = train_eta_model(df)

    # Sidebar: model diagnostics
    st.sidebar.subheader("Model diagnostics (baseline ETA)")
    if diag.get("r2") is not None:
        st.sidebar.write(f"R²: {diag['r2']:.3f}")
        st.sidebar.write(f"MAE (hrs): {diag['mae']:.3f}")
    else:
        st.sidebar.write("Not enough data to compute diagnostics.")

    # Compute composite route score for filtered df
    if "travel_time_hr" not in dff.columns:
        dff["travel_time_hr"] = dff["Distance_km"] / (dff["adjusted_speed"] + 1e-6)
    if "fuel_efficiency_kmpl" not in dff.columns:
        dff["fuel_efficiency_kmpl"] = dff.get("fuel_efficiency_kmpl", 3.7)
    dff["fuel_cost"] = (dff["Distance_km"] / (dff["fuel_efficiency_kmpl"] + 1e-9)) * FUEL_PRICE
    dff["weather_penalty"] = dff["travel_time_hr"] * (dff["weather_factor"] - 1.0)
    dff["delay_penalty"] = (dff["delay_min"] / 60.0) * 0.5
    if "toll_delay" not in dff.columns:
        dff["toll_delay"] = dff["Distance_km"].apply(lambda d: 0.1 if d < 100 else (0.2 if d < 300 else 0.3))
    dff["route_score"] = (
        dff["travel_time_hr"]
        + dff["weather_penalty"]
        + dff["delay_penalty"]
        + (dff["fuel_cost"] / 1000.0)
        + dff["toll_delay"]
    )
    dff["route_rank"] = dff["route_score"].rank(method="dense")

    # Top routes table
    st.subheader("Top Recommended Routes (by composite score)")
    top_routes = dff.sort_values("route_score").head(10).reset_index(drop=True)
    display_cols = [c for c in ["Trip_ID", "Start_Location", "End_Location", "Distance_km", "travel_time_hr", "fuel_cost", "delay_min", "Weather_Condition", "route_score", "route_rank"] if c in top_routes.columns]
    st.dataframe(top_routes[display_cols].head(10), use_container_width=True)

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Trips", dff.shape[0])
    col2.metric("Avg Delay (min)", f"{dff['delay_min'].mean():.1f}")
    col3.metric("Avg Fuel Efficiency (km/l)", f"{dff['fuel_efficiency_kmpl'].mean():.2f}")

    # Route score breakdown (top 5)
    st.subheader("Route Score Breakdown (Top 5)")
    top5 = top_routes.head(5).copy()
    if not top5.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(top5))
        travel_time = top5["travel_time_hr"].values
        weather_penalty = top5["weather_penalty"].values
        delay_penalty = top5["delay_penalty"].values
        fuel_penalty = (top5["fuel_cost"] / 1000.0).values
        toll_penalty = top5["toll_delay"].values
        bottom = np.zeros_like(travel_time)
        ax.bar(x, travel_time, label="Travel Time (hr)")
        bottom = bottom + travel_time
        ax.bar(x, weather_penalty, bottom=bottom, label="Weather Penalty")
        bottom = bottom + weather_penalty
        ax.bar(x, delay_penalty, bottom=bottom, label="Delay Penalty")
        bottom = bottom + delay_penalty
        ax.bar(x, fuel_penalty, bottom=bottom, label="Fuel Penalty (x/1000)")
        bottom = bottom + fuel_penalty
        ax.bar(x, toll_penalty, bottom=bottom, label="Toll Delay (hr)")
        ax.set_xticks(x)
        ax.set_xticklabels(top5["Trip_ID"].astype(str), rotation=45)
        ax.set_ylabel("Composite components (hours)")
        ax.legend()
        st.pyplot(fig)

    # Model validation plot
    st.subheader("Model: Predicted vs Actual (sample)")
    if diag.get("X_test") is not None:
        try:
            y_test = diag["y_test"]
            y_pred = diag["y_pred"]
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.scatter(y_test, y_pred, alpha=0.6)
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
            ax2.set_xlabel("Actual ETA (hr)")
            ax2.set_ylabel("Predicted ETA (hr)")
            ax2.set_title(f"Predicted vs Actual (R2={diag['r2']:.3f})")
            st.pyplot(fig2)
            st.write("Model MAE (hrs):", f"{diag['mae']:.3f}")
        except Exception:
            st.info("Could not render model validation plot.")
    else:
        st.info("Not enough data to show model validation (small dataset).")

    # ETA prediction small form in sidebar
    st.sidebar.subheader("Predict ETA for a new trip")
    with st.sidebar.form("eta_form"):
        in_distance = st.number_input("Distance (km)", min_value=1.0, value=100.0)
        in_speed = st.number_input("Estimated Speed (km/h)", min_value=10.0, value=70.0)
        in_weather = st.selectbox("Weather", options=["Clear", "Cloudy", "Rain", "Fog", "Snow"])
        in_traffic = st.selectbox("Traffic", options=["Low", "Medium", "High"])
        in_delay = st.number_input("Known delay (min)", min_value=0, value=0)
        submitted = st.form_submit_button("Predict ETA")
        if submitted:
            road_type = "Highway" if in_distance > 200 else "City"
            speed_limit = 80 if road_type == "Highway" else 50
            adjusted_speed = min(in_speed, speed_limit)
            base_travel_hr = in_distance / (adjusted_speed + 1e-6)
            wf_map = {"Clear": 1.0, "Cloudy": 1.05, "Rain": 1.15, "Fog": 1.25, "Snow": 1.4}
            tf_map = {"Low": 1.0, "Medium": 1.1, "High": 1.25}
            weather_factor_val = wf_map[in_weather]
            traffic_factor_val = tf_map[in_traffic]
            toll_cost_est = estimate_toll(in_distance)
            X_new = pd.DataFrame([{
                "Distance_km": in_distance,
                "adjusted_speed": adjusted_speed,
                "weather_factor": weather_factor_val,
                "traffic_factor": traffic_factor_val,
                "delay_min": in_delay,
                "toll_cost": toll_cost_est
            }])
            try:
                pred_eta = model.predict(X_new)[0]
            except Exception:
                # fallback heuristic
                pred_eta = base_travel_hr + (in_delay / 60.0) + base_travel_hr * (weather_factor_val - 1.0) + base_travel_hr * (traffic_factor_val - 1.0)
            st.sidebar.success(f"Predicted ETA: {pred_eta:.2f} hours ({pred_eta * 60:.0f} minutes)")
            st.sidebar.write(f"Base travel time (hr): {base_travel_hr:.2f}")
            st.sidebar.write(f"Adjusted speed used: {adjusted_speed} km/h")
            st.sidebar.write(f"Estimated toll cost: ₹{toll_cost_est}")

    # Export top routes
    if st.button("Export Top Routes to CSV"):
        out_path = "top_routes_export.csv"
        top_routes.to_csv(out_path, index=False)
        st.success(f"Top routes saved to {out_path}")

    # -------------------------
    # Phase 4/5: Combined Re-routing logic demo (button)
    # -------------------------
    st.markdown("## 🔄 Re-routing Decision (combined rule-based + optional live APIs + ML)")

    # Determine "current" weather & traffic for this test:
    test_weather = None
    # pick weather from weather_filter if present, else from top_routes, else "Clear"
    if len(weather_filter) > 0:
        test_weather = weather_filter[0]
    elif "Weather_Condition" in top_routes.columns and top_routes.shape[0] > 0:
        test_weather = top_routes.iloc[0].get("Weather_Condition", None)
    else:
        test_weather = "Clear"
    test_traffic = traffic_select  # manual sidebar selector
    test_delay = float(reroute_manual_delay)

    # Prepare API keys (sidebar input takes precedence, else env var)
    ow_key = openweather_key_input.strip() or os.getenv("OPENWEATHER_API_KEY", "")
    gmaps_key = google_maps_key_input.strip() or os.getenv("GOOGLE_MAPS_API_KEY", "")

    # Try live geocode, weather, directions if toggled and keys present
    live_weather_cond = None
    google_route_info = None
    ml_eta_hr = None

    # Optionally geocode origin and fetch live weather at origin
    if use_live_weather and origin_input and ow_key and requests is not None:
        # try google geocode first if gmaps_key available to get lat/lon, else skip geocode
        coords = None
        if gmaps_key:
            coords = geocode_address_google(origin_input, gmaps_key)
        # fallback: if coords not found and user typed lat,lon (rare) skip
        if coords:
            lat, lon = coords
            ow_json = get_live_weather_openweather(lat, lon, ow_key)
            if ow_json:
                live_weather_cond = parse_openweather_to_condition(ow_json) or test_weather
        else:
            # We couldn't geocode origin - skip live weather (but don't crash)
            live_weather_cond = test_weather

    # Optionally get Google directions (traffic-aware)
    if use_live_google and gmaps_key and origin_input and dest_input and requests is not None:
        google_route_info = google_directions_traffic(origin_input, dest_input, gmaps_key)

    # Optionally load ML model and predict ETA for a synthetic route (if model present)
    # We try joblib.load("truck_model.pkl") if available; if not found we skip ML
    if joblib is not None and os.path.exists("truck_model.pkl"):
        try:
            loaded = joblib.load("truck_model.pkl")
            # Build feature vector consistent with training features used earlier
            # This is a heuristic: many models require more features — adapt if necessary
            mf = {
                "Distance_km": top_routes.iloc[0]["Distance_km"] if top_routes.shape[0] > 0 else 100.0,
                "adjusted_speed": top_routes.iloc[0]["adjusted_speed"] if top_routes.shape[0] > 0 and "adjusted_speed" in top_routes.columns else 50.0,
                "weather_factor": (wf_map.get(live_weather_cond, 1.1) if live_weather_cond else 1.1),
                "delay_min": test_delay,


            "toll_cost": estimate_toll(top_routes.iloc[0]["Distance_km"]) if top_routes.shape[0] > 0 else estimate_toll(100.0),
            }
            X_m = pd.DataFrame([mf])
            # If loaded object has predict method
            if hasattr(loaded, "predict"):
                try:
                    ypred = loaded.predict(X_m)
                    # if model outputs minutes or hours depends on how trained; assume hours
                    ml_eta_hr = float(ypred[0])
                except Exception:
                    ml_eta_hr = None
        except Exception:
            ml_eta_hr = None

    # Compute the rule-based suggestion always (highest priority)
    rule_new_route, rule_reason = get_rerouted_path(current_route_id, live_weather_cond or test_weather, test_traffic, test_delay)

    # Combined decision
    final_route, final_explanation = combined_reroute_decision(
        current_route=current_route_id,
        rule_new_route=rule_new_route,
        rule_reason=rule_reason,
        google_info=google_route_info,
        ml_eta_hr=ml_eta_hr,
        manual_delay_min=test_delay,
    )

    # Button to evaluate re-route now
    if st.button("Evaluate Re-routing (using test inputs + live APIs)"):
        st.markdown("### 🔁 Re-routing Recommendation")
        # prefer final decision
        if final_route != current_route_id:
            st.error(f"⚠️ {final_explanation}")
            st.success(f"Suggested new route: **{final_route}**")
        else:
            st.info(f"✅ {final_explanation}")

    # -------------------------
    # 🔍 Re-routing Debug/Test Panel (always visible)
    # -------------------------
    st.markdown("## 🔍 Re-routing Debug Panel (Test Inputs)")
    st.write(f"**Current Route (sidebar):** {current_route_id}")
    st.write(f"**Origin input:** {origin_input or '(none)'}")
    st.write(f"**Destination input:** {dest_input or '(none)'}")
    st.write(f"**Weather for test (manual/live):** {test_weather} / live={live_weather_cond}")
    st.write(f"**Traffic for test:** {test_traffic}")
    st.write(f"**Accumulated delay (min):** {test_delay}")
    st.write("---")
    # Rule result
    debug_new_route, debug_reason = get_rerouted_path(current_route_id, live_weather_cond or test_weather, test_traffic, test_delay)
    if debug_reason:
        st.success(f"✔ Re-routing activated (rule): {debug_new_route}")
        st.write(f"**Reason:** {debug_reason}")
    else:
        st.info("ℹ️ No rule-based re-routing applied with the current test inputs.")
    # Google result
    if google_route_info is not None:
        st.write("**Google Directions (traffic-aware)**:")
        st.write(f"- Estimated distance: {google_route_info['distance_km']:.1f} km")
        dur = google_route_info.get("duration_in_traffic_min") or google_route_info.get("duration_min")
        st.write(f"- Estimated travel time (min): {dur:.0f}")
    else:
        st.write("**Google Directions:** not available / not requested")

    # ML result
    if ml_eta_hr is not None:
        st.write(f"**ML model ETA:** ~{ml_eta_hr:.2f} hrs ({ml_eta_hr*60:.0f} min)")
    else:
        st.write("**ML model ETA:** not available / not loaded")

    st.write("---")
    st.write(f"**Combined decision:** {final_route}")
    st.write(f"**Why:** {final_explanation}")

    # Footer
    st.markdown("---")
    st.caption("Smart Truck Routing — Prototype. Next steps: strengthen ML model (XGBoost), integrate route-level alternative evaluation, persist decisions and add driver alerts.")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import plotly.express as px

def analyze_hand_movement(keypoints_csv):
    """
    Analyzes hand movement from keypoint data to determine handedness and visualize movement.
    """
    # --- 1. Load Data ---
    df = pd.read_csv(keypoints_csv)

    # Keypoint IDs for wrists: 9 = left_wrist, 10 = right_wrist
    left_wrist_df = df[df['keypoint_id'] == 9]
    right_wrist_df = df[df['keypoint_id'] == 10]

    handedness_results = []

    # --- 2. Calculate Distance and Determine Handedness ---
    for person_id in df['person_id'].unique():
        # Left hand distance
        person_left = left_wrist_df[left_wrist_df['person_id'] == person_id].sort_values('frame_id')
        left_dist = np.sqrt(np.diff(person_left['x'])**2 + np.diff(person_left['y'])**2).sum()

        # Right hand distance
        person_right = right_wrist_df[right_wrist_df['person_id'] == person_id].sort_values('frame_id')
        right_dist = np.sqrt(np.diff(person_right['x'])**2 + np.diff(person_right['y'])**2).sum()

        # Determine dominant hand
        dominant_hand = 'Right' if right_dist > left_dist else 'Left'

        handedness_results.append({
            'person_id': person_id,
            'left_wrist_distance': left_dist,
            'right_wrist_distance': right_dist,
            'dominant_hand': dominant_hand
        })

    # --- 3. Save Handedness Analysis to CSV ---
    handedness_df = pd.DataFrame(handedness_results)
    handedness_df.to_csv('handedness_analysis.csv', index=False)
    print("Handedness analysis saved to handedness_analysis.csv")

    # --- 4. Create Plotly Visualization ---
    # Combine left and right wrist data for plotting
    wrists_df = pd.concat([left_wrist_df, right_wrist_df])
    wrists_df['hand'] = wrists_df['keypoint_id'].apply(lambda x: 'Left' if x == 9 else 'Right')

    # Determine axis ranges for a fixed scale
    margin = 50  # Add a margin to the plot
    x_min = df['x'].min() - margin
    x_max = df['x'].max() + margin
    y_min = df['y'].min() - margin
    y_max = df['y'].max() + margin

    fig = px.scatter(
        wrists_df,
        x='x',
        y='y',
        animation_frame='frame_id',
        animation_group='person_id',
        color='hand',
        size='confidence',
        hover_name='person_id',
        title='Hand Movement Over Time',
        labels={'x': 'X Coordinate', 'y': 'Y Coordinate', 'frame_id': 'Frame'},
        range_x=[x_min, x_max],
        range_y=[y_min, y_max]
    )

    # Improve layout and invert y-axis
    fig.update_layout(
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        legend_title="Hand",
        yaxis=dict(autorange="reversed") # Invert y-axis
    )

    # Save to HTML
    fig.write_html("hand_movement_visualization.html")
    print("Hand movement visualization saved to hand_movement_visualization.html")


if __name__ == "__main__":
    analyze_hand_movement('keypoints.csv')

import cv2
import os
import torch
from trackers import Tracker
from utils.video_utils import read_video, save_video
from team_assigner import TeamAssigner

def main():
    # ==== Step 1: Configuration ====
    input_video_path = "input/sample_video.mp4"
    model_path = "models/best.pt"
    stub_path = "stubs/track_stubs.pkl"
    output_video_path = "output_videos/output_video_using_ProvidedModel.avi" if model_path=="models/best.pt" else "output_videos/output_video_using_customModel.avi"

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # ==== Step 2: Load video frames ====
    print("Reading video...")
    video_frames = read_video(input_video_path)
    print(f"Total video_frames loaded: {len(video_frames)}")

    # ==== Step 3: Initialize tracker ====
    tracker = Tracker(model_path)

    # ==== Step 4: Run tracking ====
    print("Tracking objects...")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path=stub_path)

    # ==== Step 5: Assign Player Team ====
    team_assigner = TeamAssigner()
    team_assigner.assign_teams_color(video_frames[0], tracks["players"][0])

    for frame_num, player_tracks in enumerate(tracks["players"]):
        for player_id, track in player_tracks.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            track['team'] = team
            track['team_color'] = team_assigner.team_colors[team]

    # ==== Step 6: Annotate video_frames ====
    print("Annotating video_frames...")
    annotated_video_frames = tracker.draw_annotations(video_frames, tracks)

    # ==== Step 7: Save output ====
    print("Saving video...")
    if annotated_video_frames:
        save_video(annotated_video_frames, output_video_path)
        print(f"Annotated video saved to {output_video_path}")
    else:
        print("No video_frames to save — check video input or detection thresholds.")

if __name__ == "__main__":
    main()
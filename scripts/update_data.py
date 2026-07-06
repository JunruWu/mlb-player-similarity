from pathlib import Path
import requests

URL = "https://baseballsavant.mlb.com/leaderboard/custom?year=2026%2C2025%2C2024&type=batter&filter=&min=300&selections=pa%2Ck_percent%2Cbb_percent%2Cisolated_power%2Cxwoba%2Cavg_swing_speed%2Cfast_swing_rate%2Csquared_up_contact%2Cattack_angle%2Cexit_velocity_avg%2Claunch_angle_avg%2Csweet_spot_percent%2Cbarrel_batted_rate%2Csolidcontact_percent%2Cflareburner_percent%2Cpoorlyunder_percent%2Cpoorlytopped_percent%2Chard_hit_percent%2Cavg_best_speed%2Cavg_hyper_speed%2Cz_swing_percent%2Cz_swing_miss_percent%2Coz_swing_percent%2Coz_swing_miss_percent%2Coz_contact_percent%2Cout_zone_percent%2Ciz_contact_percent%2Cin_zone_percent%2Cedge_percent%2Cwhiff_percent%2Cswing_percent%2Cpull_percent%2Cstraightaway_percent%2Copposite_percent%2Cgroundballs_percent%2Cflyballs_percent%2Clinedrives_percent%2Cpopups_percent&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&sort=xwoba&sortDir=desc&csv=true"
output = Path("data/stats.csv")

print("Downloading latest Statcast data...")

r = requests.get(URL, timeout=60)
r.raise_for_status()

output.write_bytes(r.content)

print("Done!")

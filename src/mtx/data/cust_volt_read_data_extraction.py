import os
import glob
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd

VOLTAGE_CODES = {"51", "52", "53"}
def extract_voltage_readings(
    folder,
    max_files=500,
    patterns=["*.xml", "*.XML", "*.txt", "*.TXT"],
    recursive=False,
    keep_only_voltage=True,
    out_csv=None
):
    def iter_files(folder, patterns, recursive=False):
        for pat in patterns:
            pattern = os.path.join(folder, "**", pat) if recursive else os.path.join(folder, pat)
            for path in glob.iglob(pattern, recursive=recursive):
                if os.path.isfile(path):
                    yield path

    def parse_one_file(path):
        rows = []
        try:
            tree = ET.parse(path)
        except ET.ParseError as e: 
            print(f"[WARN] Skipping (parse error): {path} -> {e}")
            return rows
        except Exception as e:
            print(f"[WARN] Skipping (open error): {path} -> {e}")
            return rows

        root = tree.getroot()

        for ch in root.findall(".//Channels/Channel"):
            cid = ch.find("./ChannelID")
            if cid is None:
                continue
            icid = cid.get("IntervalChannelID")
            if icid is None:
                continue

            try:
                meter_id, channel_code = icid.split(":", 1)
            except ValueError:
                continue

            if keep_only_voltage:
                if channel_code not in VOLTAGE_CODES:
                    continue
            try:
                interval_len = int(ch.get("IntervalLength"))
            except (TypeError, ValueError):
                print(f"[WARN] Missing IntervalLength on {icid} in {os.path.basename(path)}; skipping channel.")
                continue

            for cis in ch.findall(".//ContiguousIntervalSet"):
                tp = cis.find("./TimePeriod")
                if tp is None:
                    continue

                start_str = tp.get("StartTime")
                if not start_str:
                    continue

                try:
                    start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                except ValueError:
                    print(f"[WARN] Bad StartTime '{start_str}' in {os.path.basename(path)}; skipping set.")
                    continue

                readings = cis.find("./Readings")
                if readings is None:
                    continue

                for i, r in enumerate(readings.findall("./Reading")):
                    v = r.get("Value")
                    if v is None:
                        continue

                    try:
                        val = float(v)
                    except ValueError:
                        continue

                    rs = start + timedelta(minutes=i * interval_len)
                    re = rs + timedelta(minutes=interval_len)

                    rows.append({
                        "SourceFile": os.path.basename(path),
                        "MeterID": meter_id,
                        "Channel": channel_code,
                        "ReadStart": rs,
                        "ReadEnd": re,
                        "VoltageReading": val,
                        "StatusRef": r.get("StatusRef"),
                    })

        return rows

    all_rows = []
    count_files = 0

    for i, path in enumerate(iter_files(folder, patterns, recursive=recursive)):
        if i >= max_files:
            break
        count_files += 1
        print(f"[INFO] Parsing {path} ...")
        rows = parse_one_file(path)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("[INFO] No rows extracted. Check folder, patterns, and channel filter.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["MeterID", "ReadStart"]).reset_index(drop=True)

    df["ReadStart"] = pd.to_datetime(df["ReadStart"])
    df["ReadEnd"] = pd.to_datetime(df["ReadEnd"])
    df = df.drop_duplicates()

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[DONE] Parsed {count_files} file(s). Wrote {len(df):,} rows to {out_csv}")

    min_start = df["ReadStart"].min()
    max_end = df["ReadEnd"].max()
    delta = max_end - min_start
    total_sec = int(delta.total_seconds())
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60

    print(f"[INFO] Time span: {min_start} → {max_end} (≈ {hrs}h {mins}m)")
    print(df.head(10).to_string(index=False))

    return df

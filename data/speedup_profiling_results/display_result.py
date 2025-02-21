#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math

def main():
    # 1. 指定你的 JSONL 文件路径
    jsonl_file = "/home/wenxuansong/chenjy/project/Consistency_LLM/data/speedup_profiling_results/speedup_calvin_profiling_results_llava-v1.5-7b-abcd-checkpoint-160000_16_1024_500_2025-02-09_11-13-41_stats.jsonl"  # 请换成你自己的文件名

    # 2. 定义累加器（accumulators）和计数器
    count = 0

    # --- fix_points 累加器 ---
    sum_fix_points = 0.0
    min_fix_points = math.inf
    max_fix_points = -math.inf

    # --- fast_forward 累加器 ---
    sum_fast_forward = 0.0
    min_fast_forward = math.inf
    max_fast_forward = -math.inf

    # --- ar_speed（先对每行取平均） 累加器 ---
    sum_ar_speed = 0.0
    min_ar_speed = math.inf
    max_ar_speed = -math.inf

    # --- jacobian_speed（先对每行取平均） 累加器 ---
    sum_jacobian_speed = 0.0
    min_jacobian_speed = math.inf
    max_jacobian_speed = -math.inf

    # --- fix_points_per_gram（先对每行取平均） 累加器 ---
    sum_fix_points_per_gram = 0.0
    min_fix_points_per_gram = math.inf
    max_fix_points_per_gram = -math.inf

    # 3. 逐行读取 JSONL 文件
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            data = json.loads(line)
            count += 1

            # ---------- fix_points ----------
            if "fix_points" in data:
                fp = data["fix_points"]
                sum_fix_points += fp
                min_fix_points = min(min_fix_points, fp)
                max_fix_points = max(max_fix_points, fp)

            # ---------- fast_forward ----------
            if "fast_forward" in data:
                ff = data["fast_forward"]
                sum_fast_forward += ff
                min_fast_forward = min(min_fast_forward, ff)
                max_fast_forward = max(max_fast_forward, ff)

            # ---------- ar_speed (列表) ----------
            # 先对当前记录的 ar_speed 列表求平均，然后做全局统计
            if "ar_speed" in data :
                # avg_ar_speed_per_record = sum(data["ar_speed"]) / len(data["ar_speed"])
                ar_speed=data["ar_speed"]
                sum_ar_speed += ar_speed
                min_ar_speed = min(min_ar_speed,ar_speed)
                max_ar_speed = max(max_ar_speed, ar_speed)

            # ---------- jacobian_speed (列表) ----------
            if "jacobian_speed" in data :
                # avg_jacobian_speed_per_record = sum(data["jacobian_speed"]) / len(data["jacobian_speed"])
                jacobian_speed = data["jacobian_speed"]
                sum_jacobian_speed += jacobian_speed
                min_jacobian_speed = min(min_jacobian_speed, jacobian_speed)
                max_jacobian_speed = max(max_jacobian_speed, jacobian_speed)

            # ---------- fix_points_per_gram (列表) ----------
            if ("fix_points_per_gram" in data 
                and isinstance(data["fix_points_per_gram"], list) 
                and len(data["fix_points_per_gram"]) > 0):
                avg_fppg_per_record = sum(data["fix_points_per_gram"]) / len(data["fix_points_per_gram"])
                sum_fix_points_per_gram += avg_fppg_per_record
                min_fix_points_per_gram = min(min_fix_points_per_gram, avg_fppg_per_record)
                max_fix_points_per_gram = max(max_fix_points_per_gram, avg_fppg_per_record)

    # 4. 计算平均值并输出结果
    if count > 0:
        # 计算平均值
        avg_fix_points = sum_fix_points / count
        avg_fast_forward = sum_fast_forward / count
        avg_ar_speed = sum_ar_speed / count
        avg_jacobian_speed = sum_jacobian_speed / count
        avg_fix_points_per_gram = sum_fix_points_per_gram / count

        # 打印结果
        print("=== Statistics across all records ===")
        print(f"Number of records: {count}\n")

        print("fix_points:")
        print(f"    Average: {avg_fix_points:.4f}")

        print("fast_forward:")
        print(f"    Average: {avg_fast_forward:.4f}")

        print("ar_speed (average per record):")
        print(f"    Average: {avg_ar_speed:.4f}, Min: {min_ar_speed:.4f}, Max: {max_ar_speed:.4f}")

        print("jacobian_speed (average per record):")
        print(f"    Average: {avg_jacobian_speed:.4f}, Min: {min_jacobian_speed:.4f}, Max: {max_jacobian_speed:.4f}")

        print("fix_points_per_gram (average per record):")
        print(f"    Average: {avg_fix_points_per_gram:.4f}, Min: {min_fix_points_per_gram:.4f}, Max: {max_fix_points_per_gram:.4f}")

    else:
        print("No valid records found in the JSONL file.")

if __name__ == "__main__":
    main()

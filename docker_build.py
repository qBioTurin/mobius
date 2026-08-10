#!/usr/bin/env python3

import argparse
import sys
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Docker image for the project")
    parser.add_argument("--tag", type=str, help="Tag for the Docker image", default="latest")
    parser.add_argument("--push", action="store_true", help="Push the Docker image after building")
    args = parser.parse_args()

    image_name = "cursecatcher/mobius"

    cl = f"docker build -t {image_name}:{args.tag} ."  # Build the Docker image
    print(f"Running command: {cl}")
    ret = subprocess.run(cl, shell=True, check=True)
    if ret.returncode != 0:
        print(f"Something went wrong: {ret.returncode}")
        sys.exit(ret.returncode)
    else:
        print("Docker image built successfully")

        if args.push:
            cl_push = f"docker push {image_name}:{args.tag}"
            print(f"Pushing Docker image with command: {cl_push}")
            ret_push = subprocess.run(cl_push, shell=True, check=True)
            if ret_push.returncode != 0:
                print(f"Failed to push Docker image: {ret_push.returncode}")
                sys.exit(ret_push.returncode)
            else:
                print("Docker image pushed successfully")

        sys.exit(0)

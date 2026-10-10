# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================
# utils.py
# Description: Utility functions
# ==============================

# Modified: retain the configuration reader used by the paper algorithms.
import json


def load_config_file(path: str) -> dict:
    """Load configuration, preserving missing-file and invalid-JSON errors."""
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)

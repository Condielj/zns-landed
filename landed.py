import hashlib
from collections import OrderedDict
from typing import Tuple, Dict, List, Union

import asyncio
import pandas as pd
from tqdm import tqdm

from dutytax import Environment, LandedCost, Item


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["hs_code"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Could not find a column called '{column}' in the input file!  Please double check."
            )

    # Optional columns
    if "ship_to_postal_code" not in df.columns:
        print(
            "Could not find a column called 'ship_to_postal_code' in the input file, US taxes will not be available."
        )
    else:
        # Cast postal codes to strings (sometimes they get read in as ints)
        df["ship_to_postal_code"] = df["ship_to_postal_code"].astype(str)

    country_columns = [
        "ship_to_country",
        "ship_from_country",
        "country_of_origin",
    ]
    bool_columns = ["is_auto_part", "is_usmca", "contains_steel_copper_or_aluminum"]

    for column in country_columns + bool_columns:
        if column not in df.columns:
            print(
                f"Could not find a column called '{column}' in input file!  Either set one per row or enter in a default value."
            )
            accepted_input = False
            while not accepted_input:
                default_value = input(
                    f"Enter a default value for '{column}' for the whole file: "
                )
                if column in country_columns and (
                    len(default_value.strip()) != 2 or default_value.isalpha()
                ):
                    print("Input must be a 2-letter country code.")
                elif column in bool_columns and default_value.strip().upper() not in (
                    "TRUE",
                    "FALSE",
                ):
                    print("Input must be a boolean value (TRUE/FALSE).")
                else:
                    accepted_input = True
                    break

            default_value = default_value.strip().upper()
            print(f"Setting {column} to {default_value} for the whole file.")
            df[column] = default_value
        else:
            # Validate values
            if column in country_columns:
                # Capitalize all values and make sure they are 2 letters
                if not all(len(val) == 2 and val.isalpha() for val in df[column]):
                    raise ValueError(
                        f"The column '{column}' must contain 2-letter country codes."
                    )
                df[column] = df[column].str.upper()
            elif column in bool_columns:
                if not all(isinstance(val, bool) for val in df[column]):
                    if not all(
                        val.upper() in ["TRUE", "FALSE", "T", "F"] for val in df[column]
                    ):
                        raise ValueError(
                            f"The column '{column}' must contain boolean values (TRUE/FALSE)."
                        )
                    df[column] = df[column].str.upper()
                else:
                    df[column] = df[column].astype(bool)

    # Set bool columns to booleans
    df["is_auto_part"] = df["is_auto_part"].astype(bool)
    df["is_usmca"] = df["is_usmca"].astype(bool)
    df["contains_steel_copper_or_aluminum"] = df[
        "contains_steel_copper_or_aluminum"
    ].astype(bool)

    return df


def csv_to_landed_costs(
    csv_file: str, postal_mode: bool = False, include_taxes: bool = True
) -> Tuple[pd.DataFrame, List[LandedCost], Dict[int, Tuple[int, int]]]:
    landed_costs = []
    df = validate_input(
        pd.read_csv(csv_file, keep_default_na=False, na_values=["", "N/A", "NULL"])
    )

    # Marshal requests
    landed_cost_columns = [
        "ship_to_country",
        "ship_from_country",
        "ship_to_postal_code",
    ]
    item_columns = [
        "country_of_origin",
        "hs_code",
        "is_auto_part",
        "is_usmca",
        "contains_steel_copper_or_aluminum",
    ]

    def _group_key(row: pd.Series, columns: List[str]) -> Tuple[str, ...]:
        return tuple(row[c] for c in columns)

    if not postal_mode:
        """
        When not in postal mode, LandedCosts can share many items since cart total will not matter.
        We need one LandedCost per unique combination of landed_cost_columns, and one Item per unique combination of landed_cost_columns + item_columns.
        We need to make sure we put the right items in the right LandedCosts.
        """
        groups = OrderedDict()
        for i, row in df.iterrows():
            gkey = _group_key(row, landed_cost_columns)
            ikey = _group_key(row, landed_cost_columns + item_columns)

            grp = groups.setdefault(
                gkey,
                {"items_by_key": OrderedDict(), "items": [], "row_to_item_idx": {}},
            )

            if ikey in grp["items_by_key"]:
                item_idx = grp["items_by_key"][ikey]
            else:
                item_kwargs = {c: row[c] for c in item_columns}
                unique_item_id = hashlib.sha256(str(item_kwargs).encode()).hexdigest()
                item_obj = Item(product_id=unique_item_id, **item_kwargs)
                item_idx = len(grp["items"])
                grp["items"].append(item_obj)
                grp["items_by_key"][ikey] = item_idx

            grp["row_to_item_idx"][i] = item_idx

        landed_costs: List[LandedCost] = []
        row_to_req_map: Dict[int, Tuple[int, int]] = {}

        for gkey, grp in groups.items():
            lc_kwargs = dict(zip(landed_cost_columns, gkey))
            lc = LandedCost(**lc_kwargs, items=grp["items"])
            lc_idx = len(landed_costs)
            landed_costs.append(lc)

            for row_idx, item_idx in grp["row_to_item_idx"].items():
                row_to_req_map[row_idx] = (lc_idx, item_idx)

        return df, landed_costs, row_to_req_map

    else:
        raise NotImplementedError("Postal mode not implemented yet")  # TODO


def landed(
    csv_file: str, postal_mode: bool = False, env: Union[Environment, str] = "prod"
) -> Tuple[pd.DataFrame, str]:
    if isinstance(env, str):
        env = Environment(env)

    df, landed_costs, row_to_req_map = csv_to_landed_costs(csv_file, postal_mode)

    async def run_all(lcs: List[LandedCost], env: Environment, concurrency: int = 20):
        sem = asyncio.Semaphore(concurrency)

        async def bound_process(idx: int, lc: LandedCost):
            async with sem:
                # return index to preserve order for row_to_req map alignment
                res = await lc.process_items(env)
                return idx, res

        tasks = [
            asyncio.create_task(bound_process(idx, lc)) for idx, lc in enumerate(lcs)
        ]

        results: List[List[Item]] = [None] * len(lcs)
        for fut in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Making landed cost requests",
            unit="req",
        ):
            idx, res = await fut
            results[idx] = res

        return results

    processed_items_by_lc: List[List[Item]] = asyncio.run(
        run_all(landed_costs, env, concurrency=20)
    )

    print(
        f"Done! Processed {len(processed_items_by_lc)} landed costs.  Preparing output..."
    )

    # Initialize output columns
    output_columns = [
        "duty_rate",
        "applied_levies",
        "tax_rate",
        "applied_taxes",
        "referenced_units_of_measure",
    ]
    for col in output_columns:
        df[col] = None

    # Fill results using the row -> (lc_idx, item_idx) map
    for row_idx, (lc_idx, item_idx) in row_to_req_map.items():
        items = (
            processed_items_by_lc[lc_idx]
            if lc_idx < len(processed_items_by_lc)
            else None
        )
        if not items or item_idx >= len(items):
            # No items for this row
            print(f"No items for row {row_idx}")
            continue

        item = items[item_idx]

        # Pull attributes
        duty_rate = item.duty_rate
        tax_rate = item.tax_rate
        applied_taxes = item.applied_taxes
        applied_levies = item.applied_levies
        referenced_units_of_measure = item.referenced_units_of_measure

        try:
            df.at[row_idx, "duty_rate"] = (
                float(duty_rate) if duty_rate is not None else None
            )
        except (TypeError, ValueError):
            df.at[row_idx, "duty_rate"] = None

        try:
            df.at[row_idx, "tax_rate"] = (
                float(tax_rate) if tax_rate is not None else None
            )
        except (TypeError, ValueError):
            df.at[row_idx, "tax_rate"] = None

        df.at[row_idx, "applied_taxes"] = applied_taxes
        df.at[row_idx, "applied_levies"] = applied_levies
        df.at[row_idx, "referenced_units_of_measure"] = referenced_units_of_measure

    out_path = csv_file.replace(".csv", "_landed_output.csv")
    df.to_csv(out_path, index=False)

    return df, out_path


if __name__ == "__main__":
    landed(csv_file="Result_4.csv", postal_mode=False, env="prod")
    print("Done")

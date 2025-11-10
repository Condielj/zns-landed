import re
import os
import time
import pandas as pd
import datetime as dt
import subprocess as sp
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Iterable, Set, Tuple
from dotenv import load_dotenv
from sqlalchemy.engine import URL
from contextlib import contextmanager
from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlmodel import SQLModel, create_engine, Session, select, func, Field


load_dotenv()


@contextmanager
def create_session(
    username: str,
    bastion_user: str,
    password: str,
    port: int,
    drivername: str = "postgresql+psycopg2",
):
    engine = bastion_engine(
        username,
        bastion_user,
        password,
        port,
        "landed_cost",
        os.getenv("DB_URL"),
        drivername,
    )
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()


def bastion_engine(
    username: str,
    bastion_user: str,
    password: str,
    port: int,
    database: str,
    host: str,
    drivername: str,
):
    sp.Popen(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=No",
            "-N",
            "-L",
            f"127.0.0.1:19001:{host}:{port}",
            f"{bastion_user}@{os.getenv('DB_BASTION_URL')}",
            "-i",
            str(Path("zdops-bastion-prod.pem")),
        ]
    )
    time.sleep(5)

    connect_args = {"sslmode": "require"}
    db_url = URL.create(
        drivername=drivername,
        username=username,
        password=password,
        host="localhost",
        port=19001,
        database=database,
    )
    return sqlalchemy_create_engine(db_url, connect_args=connect_args)


class Levy(SQLModel, table=True):
    __tablename__ = "levy"

    id: int = Field(primary_key=True)
    hs_code: str
    country: str
    country_of_origin: Optional[str] = None
    ship_from_country: Optional[str] = None
    ship_to_country: Optional[str] = None
    formula: str
    agreement: Optional[str] = None
    type: str
    condition: Optional[str] = None
    note: Optional[str] = None
    requirement: Optional[str] = None
    date_created: dt.datetime
    date_modified: Optional[dt.datetime] = None
    date_deleted: Optional[dt.datetime] = None
    levy_type: str
    levy_specific_type: Optional[str] = None
    valid_from: Optional[dt.datetime] = None
    valid_to: Optional[dt.datetime] = None
    required_measurement: Optional[str] = None


class LevyLookUp(SQLModel, table=True):
    __tablename__ = "levy_look_up"

    id: int = Field(primary_key=True)
    hs_code: str
    levy_hs_code: str
    levy_id: int
    levy_rate: float
    levy_rate_type: str
    country: str
    date_created: dt.datetime
    date_modified: Optional[dt.datetime] = None
    type: Optional[str]


class LevyCountry(SQLModel, table=True):
    __tablename__ = "levy_country"

    id: int = Field(primary_key=True)
    levy_country_alias_id: int
    country: str


def _parse_country_code_pairs(cell_value: str) -> Dict[str, str]:
    """
    Parse strigns like:
    AU = 6203.43.0004 | CA = 6203.43.0020 | DE = 6203.43.19000 | FR = 6203.43.1900 | GB = 6203.43.1900 | US = 6203.43.4500
    into a dictionary of country: hs_code pairs.
    """
    pair_regex = re.compile(r"([A-Z]{2})\s*=\s*([0-9.\-]+)")
    pairs: Dict[str, str] = {}
    chunks = [c.strip() for c in cell_value.split("|")]
    for chunk in chunks:
        if not chunk:
            continue
        m = pair_regex.search(chunk)
        if m:
            country, code = m.group(1).strip(), m.group(2).strip()
            pairs[country] = code

    return pairs


def _collect_unique_pairs(
    cells: Iterable[str],
) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    """
    From an iterable of cell strings, collect a set of unique country, code pairs for processing and a set of all country codes encountered for creating output columns
    """
    unique_pairs: Set[Tuple[str, str]] = set()
    countries: Set[str] = set()
    for cell in cells:
        pairs = _parse_country_code_pairs(cell)
        for country, code in pairs.items():
            countries.add(country)
            unique_pairs.add((country, code))

    return unique_pairs, countries


def get_duty_formula_include_agreements(
    partial_code: str, country: str, ship_from_country: str, country_of_origin: str
) -> str:
    """
    Gets the duty formula for a given HS code, importer country, ship from country, and country of origin.

    :param partial_code: The code to search.  It can be a partial HS code, as it will be used in levy_look_up to get the full HS code.
    :param country: The importing country.
    :param ship_from_country: The country where the goods are shipped from.  Sometimes required for trade agreements.
    :param country_of_origin: The country where the goods are produced.
    """

    # Get the levy_hs_code from levy_look_up
    hs_code_query = select(LevyLookUp.levy_hs_code).where(
        LevyLookUp.hs_code == partial_code, LevyLookUp.country == country
    )

    hs_code = session.exec(hs_code_query).first()
    if hs_code is None:
        raise Exception(
            f"No HS code found for {partial_code} in {country} from levy_look_up."
        )

    # Get possible duties from levy
    """
    Example Query:
        SELECT
            l.hs_code,
            coo.countries_of_origin,
            sfc.ship_from_countries,
            l.formula,
            l.agreement
        FROM
            levy l
            LEFT JOIN (
                SELECT
                    levy_country_alias_id,
                    GROUP_CONCAT(DISTINCT country_of_origin) AS countries_of_origin
                FROM
                    levy_country
                GROUP BY
                    levy_country_alias_id
            ) AS coo ON l.country_of_origin = coo.levy_country_alias_id
            LEFT JOIN (
                SELECT
                    levy_country_alias_id,
                    GROUP_CONCAT(DISTINCT country) AS ship_from_countries
                FROM
                    levy_country
                GROUP BY
                    levy_country_alias_id
            ) AS sfc ON l.ship_from_country = sfc.levy_country_alias_id
        WHERE
            l.hs_code = {hs_code}
            AND l.country = {country}
            AND l.date_deleted IS NULL
            AND l.type = 'import'
            AND l.levy_type = 'duty'
            AND l.levy_specific_type IS NULL
            AND l.condition IS NULL
            AND (coo.countries_of_origin IS NULL OR coo.countries_of_origin LIKE '%{country_of_origin}%') 
    """
    # Create the subqueries for countries of origin and ship from countries
    coo_subquery = (
        select(
            LevyCountry.levy_country_alias_id,
            func.group_concat(LevyCountry.country.distinct()).label(
                "countries_of_origin"
            ),
        )
        .group_by(LevyCountry.levy_country_alias_id)
        .subquery()
    )
    sfc_subquery = (
        select(
            LevyCountry.levy_country_alias_id,
            func.group_concat(LevyCountry.country.distinct()).label(
                "ship_from_countries"
            ),
        )
        .group_by(LevyCountry.levy_country_alias_id)
        .subquery()
    )

    # Define the main query
    duty_query = (
        select(
            Levy.hs_code,
            coo_subquery.c.countries_of_origin,
            sfc_subquery.c.ship_from_countries,
            Levy.formula,
            Levy.agreement,
        )
        .join(
            coo_subquery,
            Levy.country_of_origin == coo_subquery.c.levy_country_alias_id,
            isouter=True,
        )
        .join(
            sfc_subquery,
            Levy.ship_from_country == sfc_subquery.c.levy_country_alias_id,
            isouter=True,
        )
        .where(
            Levy.hs_code == hs_code,
            Levy.country == country,
            Levy.date_deleted == None,
            Levy.type == "import",
            Levy.levy_type == "duty",
            Levy.levy_specific_type == None,
            Levy.condition == None,
            (coo_subquery.c.countries_of_origin == None)
            | (coo_subquery.c.countries_of_origin.like(f"%{country_of_origin}%")),
        )
    )
    duty_result = session.exec(duty_query).fetchall()
    if not duty_result:
        raise Exception(f"No duty found for {hs_code} in {country} from levy.")

    # Check if there is just one duty
    if len(duty_result) == 1:
        return duty_result[0].formula
    # If there are more than one, we'll use the one with the trade agreement
    elif len(duty_result) > 1:
        for duty in duty_result:
            if (
                duty.countries_of_origin
                and ship_from_country in duty.ship_from_countries
            ):
                return duty.formula
            if duty.countries_of_origin and not duty.ship_from_country:
                return duty.formula


def get_duty_formula_no_agreements(
    input_code: str, country: str, inputs_are_full_codes: bool = True
) -> str:
    """
    Gets the duty formula for a given HS code and importer country.

    :param partial_code: The code to search.  It can be a partial HS code, as it will be used in levy_look_up to get the full HS code.
    :param country: The importing country.
    """

    if inputs_are_full_codes:
        hs_code = input_code
    else:
        # Get the levy_hs_code from levy_look_up
        hs_code_query = select(LevyLookUp.levy_hs_code).where(
            LevyLookUp.hs_code == input_code, LevyLookUp.country == country
        )

        hs_code = session.exec(hs_code_query).first()
        if hs_code is None:
            raise Exception(
                f"No HS code found for {input_code} in {country} from levy_look_up."
            )

    # Get possible duties from levy
    """
    Example Query:
        SELECT
            l.hs_code,
            l.formula
        FROM
            levy l

        WHERE
            l.hs_code = {hs_code}
            AND country_of_origin IS NULL
            AND ship_from_country IS NULL
            AND l.country = {country}
            AND l.date_deleted IS NULL
            AND l.type = 'import'
            AND l.levy_type = 'duty'
            AND l.levy_specific_type IS NULL
            AND l.condition IS NULL
    """

    # Define the main query
    duty_query = select(
        Levy.hs_code,
        Levy.formula,
    ).where(
        Levy.hs_code == hs_code,
        Levy.country_of_origin == None,
        Levy.ship_from_country == None,
        Levy.country == country,
        Levy.date_deleted == None,
        Levy.type == "import",
        Levy.levy_type == "duty",
        Levy.levy_specific_type == None,
        Levy.condition == None,
    )
    duty_result = session.exec(duty_query).fetchall()
    if not duty_result:
        print(f"No duty found for {hs_code} in {country} from levy.")
        result = f"0 % cost"
    else:
        result = duty_result[0].formula

    return result


def process_csv(
    multi_country_mode: bool,
    file_path: str,
    hs_code_column: str,
    importer_country_column: str,
    inputs_are_full_codes: bool = True,
):
    if multi_country_mode:
        # Only one input column needed
        hs_code_column = importer_country_column

    df = pd.read_csv(file_path)

    orig_len = len(df)
    df = df[df[hs_code_column].notna()]
    if len(df) != orig_len:
        print(f"Removed {orig_len-len(df)} rows that had null hs codes.")

    if not multi_country_mode:
        # Build a dictionary of unique results
        unique_combos = df[[hs_code_column, importer_country_column]].drop_duplicates()
        formula_map = {}
        with tqdm(
            total=len(unique_combos), desc="Processing unique combinations"
        ) as pbar:
            for _, row in unique_combos.iterrows():
                formula = get_duty_formula_no_agreements(
                    input_code=row[hs_code_column],
                    country=row[importer_country_column],
                    inputs_are_full_codes=inputs_are_full_codes,
                )
                formula_map[(row[hs_code_column], row[importer_country_column])] = (
                    formula
                )
                pbar.update(1)

        # Map the results back to the original DataFrame
        df["duty_formula"] = df.apply(
            lambda x: formula_map[(x[hs_code_column], x[importer_country_column])],
            axis=1,
        )
        df_out = df
    else:
        unique_pairs, all_countries = _collect_unique_pairs(
            df[importer_country_column].astype(str)
        )

        formula_map = {}
        with tqdm(
            total=len(unique_pairs), desc="Processing unique combinations"
        ) as pbar:
            for country, code in unique_pairs:
                formula = get_duty_formula_no_agreements(
                    input_code=code,
                    country=country,
                    inputs_are_full_codes=inputs_are_full_codes,
                )
                formula_map[(country, code)] = formula
                pbar.update(1)

        out_cols = [f"{country}_duty_formula" for country in sorted(all_countries)]
        row_outputs = []
        for cell in tqdm(
            df[importer_country_column].astype(str), desc="Mapping formulas to rows"
        ):
            pairs = _parse_country_code_pairs(cell)
            row_dict = {}
            for country in all_countries:
                col_name = f"{country}_duty_formula"
                if country in pairs:
                    code = pairs[country]
                    row_dict[col_name] = formula_map.get((country, code))
                else:
                    row_dict[col_name] = None
            row_outputs.append(row_dict)

        formulas_df = pd.DataFrame(row_outputs, columns=out_cols)
        df_out = pd.concat([df.reset_index(drop=True), formulas_df], axis=1)

    out_path = f"processed_{file_path}"
    df_out.to_csv(out_path, index=False)
    print(f"Output saved to {out_path}")
    return


if __name__ == "__main__":
    """
    Steps to use:

    Setup (Should only need to do this the first time you use the script):
    1. Set the environment variables for the database connection.
        Create a .env file in the same folder as the script with the following:
        DB_USER=your_db_user
        DB_PASS=your_db_password
        (Currently configured to connect to read only prod in mysql)
    2. Initialize the virtual environment.
        If already set up, run
            duty-lookup/Scripts/activate (WINDOWS)
            source duty-lookup/bin/activate (MAC)
        Or, for a new environment
            python -m, venv NAME_OF_NEW_ENV
            duty-lookup/Scripts/activate (WINDOWS)
            source duty-lookup/bin/activate (MAC)
    3. Install the required libraries.
        Run the following command in your terminal:
            pip install -r requirements.txt
            OR
            python3 -m pip install -r requirements.txt

    Running the script:
    1. Drag the file-to-be-processed into this directory.
    2. Set the column names and file name in the main section below.
    3. Run!
    4. The output file will be saved as "processed_{file_name}" in the same folder as the script.


    """
    # ================================================================
    # Set mode here!
    """
    True: expects importer_country_column to contain a list of countries and hs codes, like
    AU = 6203.42.0023 | CA = 6203.42.0081 | DE = 6203.42.35000 | FR = 6203.42.3500 | GB = 6203.42.3500 | US = 6203.42.0751
    as an example.  Each country will get its own column and rate.

    """
    multi_country_mode = True

    # Set column names here!
    hs_code_column = "HS Code"  # Column is ignored for multi_country_mode
    importer_country_column = "Ship To"

    # Set file name here!
    file_name = "WolfandBadger_TariffDatabase_V2_format_all_countries.csv"

    # Set full codes flag here! (if the CSV input codes are full hs codes, set to true)
    inputs_are_full_codes = True

    # ========================

    with create_session(
        username=os.getenv("DB_USER"),
        bastion_user=os.getenv("BASTION_USER"),
        password=os.getenv("DB_PASS"),
        port=5432,
    ) as session:
        process_csv(
            multi_country_mode=multi_country_mode,
            file_path=file_name,
            hs_code_column=hs_code_column,
            importer_country_column=importer_country_column,
            inputs_are_full_codes=inputs_are_full_codes,
        )

# db.py
import os
import json
import datetime
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime
from sqlalchemy.sql import select, delete, insert
from sqlalchemy.exc import SQLAlchemyError

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found. Set the DATABASE_URL environment variable before running.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
metadata = MetaData()

deteksi_table = Table(
    "deteksi_emosi",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, default=datetime.datetime.utcnow),
    Column("image_url", String(2048)),
    Column("hasil", Text),
)

metadata.create_all(engine)

def insert_deteksi(timestamp, image_url, hasil):
    try:
        hasil_json = json.dumps(hasil)
        query = insert(deteksi_table).values(
            timestamp=timestamp,
            image_url=image_url,
            hasil=hasil_json
        )
        with engine.begin() as conn:
            result = conn.execute(query)
            return result.inserted_primary_key[0]
    except SQLAlchemyError as e:
        print("Error insert:", e)
        return None

def list_deteksi():
    try:
        with engine.connect() as conn:
            result = conn.execute(select(deteksi_table).order_by(deteksi_table.c.timestamp.desc()))
            data = []
            for row in result:
                data.append({
                    "id": row.id,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "image_url": row.image_url,
                    "hasil": json.loads(row.hasil) if row.hasil else []
                })
            return data
    except SQLAlchemyError as e:
        print("Error list:", e)
        return []

def delete_deteksi_by_id(record_id):
    try:
        with engine.begin() as conn:
            conn.execute(delete(deteksi_table).where(deteksi_table.c.id == record_id))
            return True
    except SQLAlchemyError as e:
        print("Error delete:", e)
        return False
# database.py - sets up SQLite database using SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = "sqlite:///./stylesphere.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """gives a db session, closes it when done"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """creates the tables if they dont exist yet"""
    Base.metadata.create_all(bind=engine)

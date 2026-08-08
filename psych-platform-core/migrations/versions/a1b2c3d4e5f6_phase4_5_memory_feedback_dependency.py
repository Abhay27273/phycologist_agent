"""Phase 4+5: memory architecture, WAI-SR/SRS feedback, dependency monitoring

Adds:
  - chat_sessions.session_language (Phase 4 — language detection)
  - chat_messages.selected_move   (Phase 3 — StrategyNode move tracking)
  - mood_trajectory table         (Phase 4 — numeric time-series)
  - user_facts table              (Phase 4 — temporal entity facts)
  - session_ratings table         (Phase 5.1 — WAI-SR / SRS)
  - dependency_signals table      (Phase 5.2 — weekly dependency monitoring)

Revision ID: a1b2c3d4e5f6
Revises: 7b2047254b7f
Create Date: 2026-07-30 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "7b2047254b7f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # New columns on existing tables
    # ------------------------------------------------------------------
    op.add_column(
        "chat_sessions",
        sa.Column("session_language", sa.String(), nullable=True, server_default="en"),
    )
    op.add_column(
        "chat_messages",
        sa.Column("selected_move", sa.String(), nullable=True),
    )

    # ------------------------------------------------------------------
    # mood_trajectory
    # ------------------------------------------------------------------
    op.create_table(
        "mood_trajectory",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("session_id", sa.String(), sa.ForeignKey("chat_sessions.id"), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("mood", sa.String(), nullable=False),
        sa.Column("risk_score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_mood_trajectory_user_id", "mood_trajectory", ["user_id"])
    op.create_index("ix_mood_trajectory_session_id", "mood_trajectory", ["session_id"])

    # ------------------------------------------------------------------
    # user_facts
    # ------------------------------------------------------------------
    op.create_table(
        "user_facts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("fact_text", sa.Text(), nullable=False),
        sa.Column("entity_label", sa.String(), nullable=True),
        sa.Column("category", sa.String(), nullable=True, server_default="general"),
        sa.Column(
            "validity_start",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("validity_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_session_id", sa.String(), sa.ForeignKey("chat_sessions.id"), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True, server_default="1.0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_user_facts_user_id", "user_facts", ["user_id"])

    # ------------------------------------------------------------------
    # session_ratings  (WAI-SR + SRS)
    # ------------------------------------------------------------------
    op.create_table(
        "session_ratings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("session_id", sa.String(), sa.ForeignKey("chat_sessions.id"), nullable=False),
        sa.Column("instrument", sa.String(), nullable=False),  # "wai_sr" | "srs"
        # SRS fields
        sa.Column("srs_relationship", sa.Float(), nullable=True),
        sa.Column("srs_goals_topics", sa.Float(), nullable=True),
        sa.Column("srs_approach", sa.Float(), nullable=True),
        sa.Column("srs_overall", sa.Float(), nullable=True),
        # WAI-SR fields
        sa.Column("wai_sr_items", sa.JSON(), nullable=True),
        sa.Column("wai_sr_goals", sa.Float(), nullable=True),
        sa.Column("wai_sr_tasks", sa.Float(), nullable=True),
        sa.Column("wai_sr_bond", sa.Float(), nullable=True),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_session_ratings_user_id", "session_ratings", ["user_id"])

    # ------------------------------------------------------------------
    # dependency_signals
    # ------------------------------------------------------------------
    op.create_table(
        "dependency_signals",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("week_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("total_turn_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("night_time_turn_share", sa.Float(), nullable=True, server_default="0.0"),
        sa.Column("exclusive_reliance_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("human_support_mention_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("signal_level", sa.String(), nullable=True, server_default="normal"),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_dependency_signals_user_id", "dependency_signals", ["user_id"])
    op.create_index(
        "ix_dependency_signals_user_week",
        "dependency_signals",
        ["user_id", "week_start"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("dependency_signals")
    op.drop_table("session_ratings")
    op.drop_table("user_facts")
    op.drop_table("mood_trajectory")
    op.drop_column("chat_messages", "selected_move")
    op.drop_column("chat_sessions", "session_language")

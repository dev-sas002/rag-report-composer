# Northwind Analytics — Product Overview

## What the platform does

Northwind ingests event data from operational systems, normalises it into a
shared schema, and makes it queryable for analysts without requiring a data
engineering team in between.

## Modules

**Pipeline** connects to source systems and handles extraction, schema drift and
backfill. It supports Postgres, MySQL, Snowflake, S3 and a generic webhook
receiver.

**Modelling** provides a version-controlled transformation layer. Definitions are
written in SQL, reviewed through pull requests, and tested against sampled data
before promotion.

**Explore** is the analyst-facing query surface, with saved views, scheduled
delivery and a permissions model that follows the source system's own row-level
rules.

**Alerts** watches metric thresholds and routes notifications to email, Slack or
a webhook. Alert fatigue is managed through deduplication windows and severity
tiers.

## Deployment

The platform is offered as a managed cloud service in three regions, and as a
self-hosted deployment for customers with data residency requirements. The
self-hosted edition lags the cloud release by one minor version.

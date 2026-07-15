# 🗑️ Redundant and Obsolete Resources

During the system integration check, several redundant, duplicate, or obsolete files were identified across both codebases. These files can be safely deleted or ignored, as they are either backups or duplicated by numbered/ordered versions.

---

## 📁 ZAIgaurd-alert-engine Redundancies

### 1. `db/init/schemas.sql`
* **Redundant to:** [01_schema.sql](file:///c:/Users/saxen/Desktop/work/capstone/newMix/ZAIgaurd-alert-engine/db/init/01_schema.sql)
* **Reason:** The file `01_schema.sql` was created to replace `schemas.sql` and guarantee alphabetical execution order in the Docker Compose entrypoint (which executes `/docker-entrypoint-initdb.d` scripts in alphabetical order). 
* **Impact of removal:** Wiping this file will clean up the init folder. Currently, PostgreSQL executes both files, which works because of `IF NOT EXISTS` but generates warning notices in the database log console about existing tables.
* **Command to remove:**
  ```powershell
  Remove-Item db/init/schemas.sql
  ```

### 2. `db/init/seed.sql`
* **Redundant to:** [02_seed_data.sql](file:///c:/Users/saxen/Desktop/work/capstone/newMix/ZAIgaurd-alert-engine/db/init/02_seed_data.sql)
* **Reason:** Same as above. `02_seed_data.sql` replaces `seed.sql` to enforce correct insertion ordering (after tables are created by `01_schema.sql`).
* **Command to remove:**
  ```powershell
  Remove-Item db/init/seed.sql
  ```

---

## 📁 Zaiguard-Prototype Redundancies

### 3. `core/packets (1).py`
* **Redundant to:** [core/packets.py](file:///c:/Users/saxen/Desktop/work/capstone/newMix/Zaiguard-Prototype/core/packets.py)
* **Reason:** This is a duplicate/backup copy of `packets.py` probably created during editing. It is not referenced by any import statement in the codebase and is entirely unused.
* **Command to remove:**
  ```powershell
  Remove-Item core/packets` `(1`).py
  ```

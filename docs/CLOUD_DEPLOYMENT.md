# Deployment update

The website now uses only static hosting. See [frontend instructions](../frontend/README.md).
Do not deploy the earlier Python cloud service for this website. It is not called
by the frontend, and no visitor music is uploaded. The older cloud package remains
in the repository as unused code; existing native Python functionality is retained.

Previously uploaded test files are not automatically deleted by this change.
The browser-only app cannot access them. No new songs are sent to that service.

# Docker FAQ

**Q: How do I connect to a container from another container?**
A: When using Docker Compose, you can connect to another container using its service name as the hostname. For example, if you have a service named `lab17-redis`, you can connect to it using `redis://lab17-redis:6379/0`. Do not use `localhost` because `localhost` inside a container refers to the container itself, not the host machine or other containers.

Ensure both containers are on the same Docker network to communicate effectively.

FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      iptables ipset dnsutils curl git ca-certificates sudo \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=claude
ARG USER_UID=1000
ARG USER_GID=1001
RUN userdel -r vscode 2>/dev/null || true && \
    groupdel vscode 2>/dev/null || true && \
    groupadd -g ${USER_GID} ${USERNAME} && \
    useradd -m -s /bin/bash -u ${USER_UID} -g ${USER_GID} -G sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: /usr/local/bin/init-firewall.sh" \
      > /etc/sudoers.d/claude-firewall && \
    chmod 440 /etc/sudoers.d/claude-firewall

COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod 755 /usr/local/bin/init-firewall.sh

USER ${USERNAME}
ENV PATH="/home/${USERNAME}/.local/bin:/home/${USERNAME}/bin:${PATH}"

# Pre-create the volume mount targets owned by claude. Docker initializes
# empty named volumes by copying ownership/perms from the destination path
# in the image — without this, the volumes mount as root-owned and claude
# (UID 1000) can't write to them.
RUN mkdir -p /home/${USERNAME}/.cache/uv /home/${USERNAME}/.claude

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN curl -fsSL https://claude.ai/install.sh | bash

WORKDIR /workspaces/JacobianODE

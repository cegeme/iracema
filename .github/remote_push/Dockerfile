FROM alpine:3.10

RUN apk update && apk upgrade && \
    apk add --no-cache openssh git

COPY push.sh /push.sh

ENTRYPOINT ["/push.sh"]

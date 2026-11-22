LOCAL_LIMITS = {'run_request_limit', 'rolling_24h_request_limit', 'not_sent_quota'}
VALIDATION = {'invalid_json_response', 'invalid_semantic_schema', 'invalid_response_envelope', 'incomplete_or_refused_response'}


def event_kind(event):
    code = event.get('error')
    if not code:
        return 'cache' if event.get('status') == 'gemini' and event.get('cached') else 'success' if event.get('status') == 'gemini' else 'other'
    if code in LOCAL_LIMITS:
        return 'limit_pause'
    if code in VALIDATION:
        return 'validation_error'
    if code.startswith('http_') or code == 'network_outcome_unknown_no_automatic_retry':
        return 'api_error'
    if code == 'uncertain_previous_request_no_automatic_retry':
        return 'safety_pause'
    return 'other_error'


def event_metrics(stats):
    calls = stats.get('calls', [])
    kinds = [event_kind(c) for c in calls]
    result = {name: kinds.count(kind) for name, kind in {
        'api_errors':'api_error', 'validation_errors':'validation_error', 'limit_pauses':'limit_pause',
        'safety_pauses':'safety_pause', 'other_errors':'other_error'}.items()}
    result['processing_errors'] = result['api_errors'] + result['validation_errors'] + result['other_errors']
    result['original_error_events'] = sum(bool(c.get('error')) for c in calls)
    latest = {c['row']: c for c in calls}
    result['accepted_responses'] = sum(c.get('status') == 'gemini' and not c.get('error') for c in latest.values())
    result['accepted_cached_responses'] = sum(c.get('status') == 'gemini' and not c.get('error') and bool(c.get('cached')) for c in latest.values())
    result['accepted_new_responses'] = result['accepted_responses'] - result['accepted_cached_responses']
    return result


MESSAGES = {
    'missing_or_invalid_api_key':'Не удалось подключиться к Gemini. Проверьте настройки подключения.',
    'http_401':'Не удалось подключиться к Gemini. Проверьте настройки подключения.',
    'http_403':'Сервис Gemini не разрешил доступ. Проверьте настройки подключения и доступность сервиса в вашем регионе.',
    'http_404':'Выбранная модель недоступна. Измените модель в настройках нового эксперимента.',
    'http_400':'Сервис не принял настройки обработки. Проверьте модель и параметры подключения.',
    'http_429':'Достигнут лимит запросов. Продолжите обработку после восстановления квоты.',
    'http_503':'Сервис Gemini временно недоступен. Продолжите обработку позже.',
    'run_request_limit':'Достигнут установленный для эксперимента лимит запросов.',
    'rolling_24h_request_limit':'Достигнут установленный суточный лимит. Продолжите после его восстановления.',
    'network_outcome_unknown_no_automatic_retry':'Нет соединения с сервисом Gemini. Исход последнего запроса неизвестен; повторная отправка заблокирована, чтобы избежать повторного расхода квоты.',
    'uncertain_previous_request_no_automatic_retry':'Исход предыдущего запроса неизвестен. Его повторная отправка заблокирована. Обратитесь к администратору, сохранив журнал.',
    'free_tier_not_confirmed':'Подтвердите доступную бесплатную квоту перед запуском Gemini.',
    'invalid_numeric_configuration':'Некорректные настройки подключения. Проверьте числовые параметры.',
    'invalid_request_limits':'Проверьте допустимые значения ограничений запросов и времени ожидания.',
    'invalid_model':'Название модели некорректно. Проверьте настройки подключения.',
}
MESSAGES.update({code:'Не удалось обработать ответ модели. Ответ сохранён, но не считается успешным результатом Gemini.' for code in VALIDATION})


def user_message(code):
    if code in MESSAGES:
        return MESSAGES[code]
    if code.startswith('http_'):
        return 'Сервис Gemini сообщил об ошибке. Подробности сохранены в журнале.'
    # Do not expose arbitrary exceptions, credential-bearing strings or paths.
    if any('\u0400' <= c <= '\u04ff' for c in code) and not any(x in code for x in ('.env', 'API_KEY', 'AIza', 'Bearer', 'Traceback', 'SHA-256')):
        return code
    return 'Не удалось выполнить операцию. Проверьте выбранные данные и настройки подключения. Подробности доступны в технической информации.'
